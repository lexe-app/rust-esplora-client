// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Asynchronous Esplora Client
//!
//! This module implements [`AsyncClient`], an asynchronous HTTP client for
//! interacting with an [Esplora] server by way of [`reqwest`].
//!
//! Use this client from async applications and libraries. Each method returns a
//! future that sends the request, waits for the response, and decodes the body
//! into the requested type.
//!
//! The client is configured through [`Builder`], including the
//! base URL, proxy, socket timeout, custom headers, retry count, and maximum
//! number of cached connections. Retry sleeping is abstracted through
//! [`Sleeper`], so runtimes other than Tokio can provide their own sleep
//! implementation.
//!
//! # Example
//!
//! ```rust,ignore
//! # use esplora_client::{Builder, r#async::AsyncClient};
//! # async fn example() -> Result<(), esplora_client::Error> {
//!
//! let client = Builder::new("https://mempool.space/api").build_async()?;
//! let height = client.get_height().await?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! [Esplora]: https://github.com/Blockstream/esplora/blob/master/API.md

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use std::time::{Duration, Instant};

use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::consensus::{deserialize, serialize, Decodable};
use bitcoin::hashes::{sha256, Hash};
use bitcoin::hex::{DisplayHex, FromHex};
use bitcoin::{Address, Amount, Block, BlockHash, FeeRate, MerkleBlock, Script, Transaction, Txid};

use tracing::debug;

use crate::{
    sat_per_vbyte_to_feerate, AddressStats, BlockInfo, BlockStatus, Builder, Error, EsploraTx,
    HttpResponse, MempoolRecentTx, MempoolStats, MerkleProof, OutputStatus, ScriptHashStats,
    SubmitPackageResult, TxStatus, Utxo, BASE_BACKOFF_MILLIS,
};

#[allow(deprecated)]
use crate::BlockSummary;

/// An async client for interacting with an Esplora API server.
///
/// Use [`Builder`] to construct an instance of this client. The client stores
/// the server base URL and request configuration, then exposes convenience
/// methods for the transaction, block, address, scripthash, fee-estimate, and
/// mempool endpoints.
///
/// The generic parameter `S` determines the asynchronous runtime used for
/// sleeping between retries. Defaults to the Tokio-backed [`DefaultSleeper`].
///
/// # Retries
///
/// Failed requests are automatically retried up to `max_retries` times
/// (configured via [`Builder`]) with exponential backoff, but only for
/// retryable HTTP status codes. See [`crate::RETRYABLE_ERROR_CODES`] for the
/// full list.
#[derive(Clone)]
pub struct AsyncClient<S = DefaultSleeper> {
    /// The URL of the Esplora server.
    url: String,
    /// Maximum number of retry attempts for retryable responses.
    max_retries: usize,
    /// The inner HTTP [`reqwest::Client`] which caches connections.
    client: reqwest::Client,
    /// Marker for the sleeper implementation.
    marker: PhantomData<S>,
}

impl<S: Sleeper> AsyncClient<S> {
    /// Build an [`AsyncClient`] from a [`Builder`].
    ///
    /// Configures the underlying [`reqwest::Client`] with the proxy, timeout,
    /// headers, and connection limit specified in the [`Builder`].
    /// No network request is made until a client method is awaited.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the HTTP client fails to build,
    /// or if any of the provided header names or values are invalid.
    pub fn from_builder(builder: Builder) -> Result<Self, Error> {
        let mut client_builder = reqwest::Client::builder();

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(proxy) = &builder.proxy {
            client_builder = client_builder.proxy(reqwest::Proxy::all(proxy)?);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(timeout) = builder.timeout {
            client_builder = client_builder.timeout(timeout);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            client_builder = client_builder.pool_max_idle_per_host(builder.max_connections);
        }

        if !builder.headers.is_empty() {
            let mut headers = reqwest::header::HeaderMap::new();
            for (k, v) in builder.headers {
                let header_name =
                    reqwest::header::HeaderName::from_lowercase(k.to_lowercase().as_bytes())
                        .map_err(|_| Error::InvalidHttpHeaderName(k))?;
                let header_value = reqwest::header::HeaderValue::from_str(&v)
                    .map_err(|_| Error::InvalidHttpHeaderValue(v))?;
                headers.insert(header_name, header_value);
            }
            client_builder = client_builder.default_headers(headers);
        }

        Ok(AsyncClient {
            url: builder.base_url,
            max_retries: builder.max_retries,
            client: client_builder.build()?,
            marker: PhantomData,
        })
    }

    /// Build an [`AsyncClient`] from a base `url` and an existing
    /// [`reqwest::Client`].
    ///
    /// Lets callers inject a client with a custom configuration, e.g. a
    /// specific TLS setup via `reqwest::ClientBuilder::use_preconfigured_tls`.
    pub fn from_client(url: String, client: reqwest::Client) -> Self {
        AsyncClient {
            url,
            max_retries: crate::DEFAULT_MAX_RETRIES,
            client,
            marker: PhantomData,
        }
    }

    /// Return the base URL of the Esplora server this client connects to.
    ///
    /// The returned value is the exact string provided to [`Builder::new`].
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Return the underlying HTTP [`reqwest::Client`].
    ///
    /// This can be useful for callers that need access to shared connection
    /// state managed by the HTTP client.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Sends a single GET request to `path`.
    async fn send_get(&self, path: &str) -> Result<HttpResponse, Error> {
        let url = format!("{}{}", self.url, path);
        let response = self.client.get(url).send().await?;
        HttpResponse::from_reqwest(response).await
    }

    /// Sends a single POST request to `path` with `body` and query parameters.
    async fn send_post(
        &self,
        path: &str,
        body: Vec<u8>,
        query_params: Option<HashSet<(&str, String)>>,
    ) -> Result<HttpResponse, Error> {
        let url = format!("{}{}", self.url, path);
        let mut request = self.client.post(url).body(body);

        for (key, value) in query_params.unwrap_or_default() {
            request = request.query(&[(key, value)]);
        }

        let response = request.send().await?;
        HttpResponse::from_reqwest(response).await
    }

    /// Sends a GET request to `path`, retrying on retryable status codes
    /// with exponential backoff until [`AsyncClient::max_retries`] is reached.
    ///
    /// Returns an [`Error::HttpResponse`] on a non-success status code.
    async fn get_with_retry(&self, path: &str) -> Result<HttpResponse, Error> {
        let mut delay = BASE_BACKOFF_MILLIS;
        let mut attempts = 0;

        loop {
            let start = Instant::now();
            let result = self.send_get(path).await;
            self.log_result(path, "GET", &result, start);
            let response = result?;

            if attempts < self.max_retries && response.is_retryable() {
                S::sleep(delay).await;
                attempts += 1;
                delay *= 2;
            } else {
                return response.error_for_status();
            }
        }
    }

    /// Lexe patch: Log every request, including failures, so we know:
    ///
    /// 1) which endpoints we're calling,
    /// 2) how many requests we're making,
    /// 3) which services these requests are sent to,
    /// 4) how long each request takes,
    /// 5) how large the response is, and
    /// 6) how long failed requests took to fail.
    ///
    /// TODO(max): Eventually this should be done with metrics, not logs.
    fn log_result(
        &self,
        path: &str,
        method: &str,
        result: &Result<HttpResponse, Error>,
        start: Instant,
    ) {
        let base_url = &self.url;
        // For user privacy, log only the first path segment, which identifies
        // the endpoint without the user-specific parameters that follow.
        let endpoint = first_path_segment(path);
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            // "GET https://blockstream.info/api/address
            //  => 200, HTTP/2.0, 3ms, 1.2KiB"
            Ok(response) => {
                let status = response.status;
                let version = response.version;
                let size_kib = response.body.len() as f64 / 1024.0;
                debug!(
                    "{method} {base_url}{endpoint} \
                     => {status}, {version}, {duration_ms:.3}ms, {size_kib:.3}KiB"
                );
            }
            // "GET https://blockstream.info/api/address
            //  => error, 30000.000ms: error sending request"
            Err(error) => debug!(
                "{method} {base_url}{endpoint} \
                 => error, {duration_ms:.3}ms: {error}"
            ),
        }
    }

    /// Makes a GET request to `path`, deserializing the response body as raw
    /// bytes into `T` using [`bitcoin::consensus::Decodable`].
    ///
    /// Use this for endpoints that return raw binary Bitcoin data.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails or deserialization fails.
    async fn get_response<T: Decodable>(&self, path: &str) -> Result<T, Error> {
        let response = self.get_with_retry(path).await?;
        Ok(deserialize::<T>(&response.body)?)
    }

    /// Makes a GET request to `path`, returning `None` on a 404 response.
    ///
    /// Delegates to [`Self::get_response`]. See its documentation for details.
    async fn get_opt_response<T: Decodable>(&self, path: &str) -> Result<Option<T>, Error> {
        match self.get_response::<T>(path).await {
            Ok(res) => Ok(Some(res)),
            Err(Error::HttpResponse { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Makes a GET request to `path`, deserializing the response body as JSON
    /// into `T` using [`serde::de::DeserializeOwned`].
    ///
    /// Use this for endpoints that return Esplora-specific JSON types, as
    /// defined in [`crate::api`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails or JSON deserialization fails.
    async fn get_response_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<T, Error> {
        let response = self.get_with_retry(path).await?;
        response.json::<T>()
    }

    /// Makes a GET request to `path`, returning `None` on a 404 response.
    ///
    /// Delegates to [`Self::get_response_json`]. See its documentation for details.
    async fn get_opt_response_json<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
    ) -> Result<Option<T>, Error> {
        match self.get_response_json(url).await {
            Ok(res) => Ok(Some(res)),
            Err(Error::HttpResponse { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Makes a GET request to `path`, deserializing the hex-encoded response
    /// body into `T` using [`bitcoin::consensus::Decodable`].
    ///
    /// Use this for endpoints that return hex-encoded Bitcoin data.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails, hex decoding fails,
    /// or consensus deserialization fails.
    async fn get_response_hex<T: Decodable>(&self, path: &str) -> Result<T, Error> {
        let response = self.get_with_retry(path).await?;
        let hex_str = response.as_str()?;
        Ok(deserialize(&Vec::from_hex(hex_str)?)?)
    }

    /// Makes a GET request to `path`, returning `None` on a 404 response.
    ///
    /// Delegates to [`Self::get_response_hex`]. See its documentation for details.
    async fn get_opt_response_hex<T: Decodable>(&self, path: &str) -> Result<Option<T>, Error> {
        match self.get_response_hex(path).await {
            Ok(res) => Ok(Some(res)),
            Err(Error::HttpResponse { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Makes a GET request to `path`, returning the response body as a [`String`].
    ///
    /// Use this for endpoints that return plain text data that needs further
    /// parsing downstream.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails.
    async fn get_response_text(&self, path: &str) -> Result<String, Error> {
        let response = self.get_with_retry(path).await?;
        Ok(response.as_str()?.to_string())
    }

    /// Makes a GET request to `path`, returning `None` on a 404 response.
    ///
    /// Delegates to [`Self::get_response_text`]. See its documentation for details.
    async fn get_opt_response_text(&self, path: &str) -> Result<Option<String>, Error> {
        match self.get_response_text(path).await {
            Ok(s) => Ok(Some(s)),
            Err(Error::HttpResponse { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Make an HTTP POST request to `path` with `body`.
    ///
    /// Configures query parameters, if any, and returns the raw response after
    /// checking the HTTP status code.
    ///
    /// # Errors
    ///
    /// This function will return an error either from the HTTP client, or the
    /// response's [`serde_json`] deserialization.
    async fn post_request_bytes<T: Into<Vec<u8>>>(
        &self,
        path: &str,
        body: T,
        query_params: Option<HashSet<(&str, String)>>,
    ) -> Result<HttpResponse, Error> {
        let start = Instant::now();
        let result = self.send_post(path, body.into(), query_params).await;
        self.log_result(path, "POST", &result, start);
        result?.error_for_status()
    }

    /// Get a raw [`Transaction`] given its [`Txid`].
    ///
    /// Returns `None` if the transaction is not found.
    pub async fn get_tx(&self, txid: &Txid) -> Result<Option<Transaction>, Error> {
        self.get_opt_response(&format!("/tx/{txid}/raw")).await
    }

    /// Get a raw [`Transaction`] given its [`Txid`].
    ///
    /// Returns an [`Error::TransactionNotFound`] if the transaction is not found.
    /// Prefer [`Self::get_tx`] if you want to handle the not-found case explicitly.
    pub async fn get_tx_no_opt(&self, txid: &Txid) -> Result<Transaction, Error> {
        match self.get_tx(txid).await {
            Ok(Some(tx)) => Ok(tx),
            Ok(None) => Err(Error::TransactionNotFound(*txid)),
            Err(e) => Err(e),
        }
    }

    /// Get the [`Txid`] of the transaction at position `index` within the
    /// block identified by `block_hash`.
    ///
    /// Returns `None` if the block or index is not found.
    pub async fn get_txid_at_block_index(
        &self,
        block_hash: &BlockHash,
        index: usize,
    ) -> Result<Option<Txid>, Error> {
        match self
            .get_opt_response_text(&format!("/block/{block_hash}/txid/{index}"))
            .await?
        {
            Some(s) => Ok(Some(Txid::from_str(&s).map_err(Error::HexToArray)?)),
            None => Ok(None),
        }
    }

    /// Get the confirmation status of a [`Transaction`] given its [`Txid`].
    ///
    /// Returns a [`TxStatus`] containing whether the transaction is confirmed,
    /// and if so, the block height, hash, and timestamp it was confirmed in.
    pub async fn get_tx_status(&self, txid: &Txid) -> Result<TxStatus, Error> {
        self.get_response_json(&format!("/tx/{txid}/status")).await
    }

    /// Get an [`EsploraTx`] given its [`Txid`].
    ///
    /// Unlike [`Self::get_tx`], returns the Esplora-specific [`EsploraTx`]
    /// type, which includes additional metadata such as confirmation status,
    /// fee, and weight. Returns `None` if the transaction is not found.
    pub async fn get_tx_info(&self, txid: &Txid) -> Result<Option<EsploraTx>, Error> {
        self.get_opt_response_json(&format!("/tx/{txid}")).await
    }

    /// Get the spend status of all outputs in a [`Transaction`], given its [`Txid`].
    ///
    /// Returns a [`Vec`] of [`OutputStatus`], one per output, ordered as they
    /// appear in the [`Transaction`].
    pub async fn get_tx_outspends(&self, txid: &Txid) -> Result<Vec<OutputStatus>, Error> {
        self.get_response_json(&format!("/tx/{txid}/outspends"))
            .await
    }

    /// Get the [`BlockHeader`] of a [`Block`] given its [`BlockHash`].
    pub async fn get_header_by_hash(&self, block_hash: &BlockHash) -> Result<BlockHeader, Error> {
        self.get_response_hex(&format!("/block/{block_hash}/header"))
            .await
    }

    /// Get the [`BlockStatus`] of a [`Block`] given its [`BlockHash`].
    ///
    /// Returns a [`BlockStatus`] indicating whether this [`Block`] is part of
    /// the best chain, its height, and the [`BlockHash`] of the next [`Block`],
    /// if any.
    pub async fn get_block_status(&self, block_hash: &BlockHash) -> Result<BlockStatus, Error> {
        self.get_response_json(&format!("/block/{block_hash}/status"))
            .await
    }

    /// Get the full [`Block`] with the given [`BlockHash`].
    ///
    /// Returns `None` if the [`Block`] is not found.
    pub async fn get_block_by_hash(&self, block_hash: &BlockHash) -> Result<Option<Block>, Error> {
        self.get_opt_response(&format!("/block/{block_hash}/raw"))
            .await
    }

    /// Get a Merkle inclusion proof for a [`Transaction`] given its [`Txid`].
    ///
    /// Returns a [`MerkleProof`] that can be used to verify the transaction's
    /// inclusion in a block. Returns `None` if the transaction is not found or
    /// is unconfirmed.
    pub async fn get_merkle_proof(&self, tx_hash: &Txid) -> Result<Option<MerkleProof>, Error> {
        self.get_opt_response_json(&format!("/tx/{tx_hash}/merkle-proof"))
            .await
    }

    /// Get a [`MerkleBlock`] inclusion proof for a [`Transaction`] given its [`Txid`].
    ///
    /// Returns `None` if the transaction is not found or is unconfirmed.
    pub async fn get_merkle_block(&self, tx_hash: &Txid) -> Result<Option<MerkleBlock>, Error> {
        self.get_opt_response_hex(&format!("/tx/{tx_hash}/merkleblock-proof"))
            .await
    }

    /// Get the spend status of a specific output, identified by its [`Txid`]
    /// and output index.
    ///
    /// Returns an [`OutputStatus`] indicating whether the output has been
    /// spent, and if so, by which transaction. Returns `None` if not found.
    pub async fn get_output_status(
        &self,
        txid: &Txid,
        index: u64,
    ) -> Result<Option<OutputStatus>, Error> {
        self.get_opt_response_json(&format!("/tx/{txid}/outspend/{index}"))
            .await
    }

    /// Broadcast a [`Transaction`] to the Esplora server.
    ///
    /// The transaction is serialized and sent as a hex-encoded string.
    /// Returns the [`Txid`] of the broadcasted transaction.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails or the server rejects the transaction.
    pub async fn broadcast(&self, transaction: &Transaction) -> Result<Txid, Error> {
        let body = serialize::<Transaction>(transaction).to_lower_hex_string();
        let response = self.post_request_bytes("/tx", body, None).await?;
        let txid = Txid::from_str(response.as_str()?).map_err(Error::HexToArray)?;
        Ok(txid)
    }

    /// Broadcast a package of [`Transaction`]s to the Esplora server.
    ///
    /// Returns a [`SubmitPackageResult`] containing the result for each
    /// transaction in the package, keyed by [`bitcoin::Wtxid`].
    ///
    /// Optionally, `maxfeerate` and `maxburnamount` can be provided to reject
    /// transactions that exceed these thresholds.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the request fails or the server rejects the package.
    pub async fn submit_package(
        &self,
        transactions: &[Transaction],
        maxfeerate: Option<FeeRate>,
        maxburnamount: Option<Amount>,
    ) -> Result<SubmitPackageResult, Error> {
        let serialized_txs = transactions
            .iter()
            .map(|tx| serialize_hex(&tx))
            .collect::<Vec<_>>();

        let mut params = HashSet::<(&str, String)>::new();

        // Esplora expects `maxfeerate` in sats/vB.
        if let Some(maxfeerate) = maxfeerate {
            params.insert(("maxfeerate", maxfeerate.to_sat_per_vb_ceil().to_string()));
        }
        // Esplora expects `maxburnamount` in BTC.
        if let Some(maxburnamount) = maxburnamount {
            params.insert(("maxburnamount", maxburnamount.to_btc().to_string()));
        }

        let response = self
            .post_request_bytes(
                "/txs/package",
                serde_json::to_string(&serialized_txs).map_err(Error::SerdeJson)?,
                Some(params),
            )
            .await?;

        let result = response.json::<SubmitPackageResult>()?;

        Ok(result)
    }

    /// Get the block height of the current blockchain tip.
    pub async fn get_height(&self) -> Result<u32, Error> {
        self.get_response_text("/blocks/tip/height")
            .await
            .map(|height| u32::from_str(&height).map_err(Error::Parsing))?
    }

    /// Get the [`BlockHash`] of the current blockchain tip.
    pub async fn get_tip_hash(&self) -> Result<BlockHash, Error> {
        self.get_response_text("/blocks/tip/hash")
            .await
            .map(|block_hash| BlockHash::from_str(&block_hash).map_err(Error::HexToArray))?
    }

    /// Get the [`BlockHash`] of a [`Block`] given its height.
    pub async fn get_block_hash(&self, block_height: u32) -> Result<BlockHash, Error> {
        self.get_response_text(&format!("/block-height/{block_height}"))
            .await
            .map(|block_hash| BlockHash::from_str(&block_hash).map_err(Error::HexToArray))?
    }

    /// Get statistics about an [`Address`].
    ///
    /// Returns an [`AddressStats`] containing confirmed and mempool transaction
    /// summaries for the given address, including funded and spent output
    /// counts and their total values.
    pub async fn get_address_stats(&self, address: &Address) -> Result<AddressStats, Error> {
        let path = format!("/address/{address}");
        self.get_response_json(&path).await
    }

    /// Get statistics about a [`Script`] hash's confirmed and mempool transactions.
    ///
    /// Returns a [`ScriptHashStats`] containing transaction summaries for the
    /// SHA256 hash of the given [`Script`].
    pub async fn get_scripthash_stats(&self, script: &Script) -> Result<ScriptHashStats, Error> {
        let script_hash = sha256::Hash::hash(script.as_bytes());
        let path = format!("/scripthash/{script_hash}");
        self.get_response_json(&path).await
    }

    /// Get confirmed transaction history for an [`Address`], sorted newest first.
    ///
    /// Returns up to 50 mempool transactions plus the first 25 confirmed transactions.
    /// To paginate, pass the [`Txid`] of the last transaction seen in the
    /// previous response as `last_seen`.
    pub async fn get_address_txs(
        &self,
        address: &Address,
        last_seen: Option<Txid>,
    ) -> Result<Vec<EsploraTx>, Error> {
        let path = match last_seen {
            Some(last_seen) => format!("/address/{address}/txs/chain/{last_seen}"),
            None => format!("/address/{address}/txs"),
        };

        self.get_response_json(&path).await
    }

    /// Get unconfirmed mempool [`EsploraTx`]s for an [`Address`], sorted newest first.
    pub async fn get_mempool_address_txs(
        &self,
        address: &Address,
    ) -> Result<Vec<EsploraTx>, Error> {
        let path = format!("/address/{address}/txs/mempool");

        self.get_response_json(&path).await
    }

    /// Get confirmed transaction history for a [`Script`] hash, sorted newest first.
    ///
    /// Returns 25 transactions per page. To paginate, pass the [`Txid`] of the
    /// last transaction seen in the previous response as `last_seen`.
    pub async fn get_scripthash_txs(
        &self,
        script: &Script,
        last_seen: Option<Txid>,
    ) -> Result<Vec<EsploraTx>, Error> {
        let script_hash = sha256::Hash::hash(script.as_bytes());
        let path = match last_seen {
            Some(last_seen) => format!("/scripthash/{script_hash:x}/txs/chain/{last_seen}"),
            None => format!("/scripthash/{script_hash:x}/txs"),
        };

        self.get_response_json(&path).await
    }

    /// Get unconfirmed mempool [`EsploraTx`]s for a [`Script`] hash, sorted newest first.
    pub async fn get_mempool_scripthash_txs(
        &self,
        script: &Script,
    ) -> Result<Vec<EsploraTx>, Error> {
        let script_hash = sha256::Hash::hash(script.as_bytes());
        let path = format!("/scripthash/{script_hash:x}/txs/mempool");

        self.get_response_json(&path).await
    }

    /// Get global statistics about the mempool.
    ///
    /// Returns a [`MempoolStats`] containing the transaction count, total
    /// virtual size, total fees, and fee rate histogram.
    pub async fn get_mempool_stats(&self) -> Result<MempoolStats, Error> {
        self.get_response_json("/mempool").await
    }

    /// Get the last 10 [`MempoolRecentTx`]s to enter the mempool.
    pub async fn get_mempool_recent_txs(&self) -> Result<Vec<MempoolRecentTx>, Error> {
        self.get_response_json("/mempool/recent").await
    }

    /// Get the full list of [`Txid`]s in the mempool.
    ///
    /// The order of the [`Txid`]s is arbitrary.
    pub async fn get_mempool_txids(&self) -> Result<Vec<Txid>, Error> {
        self.get_response_json("/mempool/txids").await
    }

    /// Get fee estimates for a range of confirmation targets.
    ///
    /// Returns a [`HashMap`] where the key is the confirmation target in blocks
    /// and the value is the estimated [`FeeRate`].
    pub async fn get_fee_estimates(&self) -> Result<HashMap<u16, FeeRate>, Error> {
        let estimates_raw: HashMap<u16, f64> = self.get_response_json("/fee-estimates").await?;
        let estimates = sat_per_vbyte_to_feerate(estimates_raw);

        Ok(estimates)
    }

    /// Get a [`BlockInfo`] summary for the [`Block`] with the given [`BlockHash`].
    ///
    /// [`BlockInfo`] includes metadata such as the height, timestamp,
    /// [`Transaction`] count, size, and [`Weight`](bitcoin::Weight).
    ///
    /// This method does not return the full [`Block`].
    pub async fn get_block_info(&self, blockhash: &BlockHash) -> Result<BlockInfo, Error> {
        let path = format!("/block/{blockhash}");

        self.get_response_json(&path).await
    }

    /// Get all [`Txid`]s of [`Transaction`]s included in the [`Block`] with the
    /// given [`BlockHash`].
    pub async fn get_block_txids(&self, blockhash: &BlockHash) -> Result<Vec<Txid>, Error> {
        let path = format!("/block/{blockhash}/txids");

        self.get_response_json(&path).await
    }

    /// Get up to 25 [`EsploraTx`]s from the [`Block`] with the given
    /// [`BlockHash`], starting at `start_index`.
    ///
    /// If `start_index` is `None`, starts from the first transaction (index 0).
    ///
    /// Note that `start_index` must be a multiple of 25, otherwise the server
    /// will return an error.
    pub async fn get_block_txs(
        &self,
        blockhash: &BlockHash,
        start_index: Option<u32>,
    ) -> Result<Vec<EsploraTx>, Error> {
        let path = match start_index {
            None => format!("/block/{blockhash}/txs"),
            Some(start_index) => format!("/block/{blockhash}/txs/{start_index}"),
        };

        self.get_response_json(&path).await
    }

    /// Get [`BlockInfo`] summaries for recent [`Block`]s.
    ///
    /// If `height` is `Some(h)`, returns blocks starting from height `h`.
    /// If `height` is `None`, returns blocks starting from the current tip.
    ///
    /// The maximum number of summaries returned depends on the backend itself:
    /// Esplora returns `10` while [mempool.space](https://mempool.space/docs/api) returns `15`.
    #[allow(deprecated)]
    #[deprecated(since = "0.13.0", note = "use `get_block_infos` instead")]
    pub async fn get_blocks(&self, height: Option<u32>) -> Result<Vec<BlockSummary>, Error> {
        let path = match height {
            Some(height) => format!("/blocks/{height}"),
            None => "/blocks".to_string(),
        };
        let blocks: Vec<BlockSummary> = self.get_response_json(&path).await?;
        if blocks.is_empty() {
            return Err(Error::InvalidResponse);
        }
        Ok(blocks)
    }

    /// Get [`BlockInfo`] summaries for recent [`Block`]s.
    ///
    /// If `height` is `Some(h)`, returns blocks starting from height `h`.
    /// If `height` is `None`, returns blocks starting from the current tip.
    ///
    /// The maximum number of summaries returned depends on the backend itself:
    /// Esplora returns `10` while [mempool.space](https://mempool.space/docs/api) returns `15`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidResponse`] if the server returns an empty list.
    ///
    /// This method does not return the full [`Block`].
    pub async fn get_block_infos(&self, height: Option<u32>) -> Result<Vec<BlockInfo>, Error> {
        let path = match height {
            Some(height) => format!("/blocks/{height}"),
            None => "/blocks".to_string(),
        };
        let blocks: Vec<BlockInfo> = self.get_response_json(&path).await?;
        if blocks.is_empty() {
            return Err(Error::InvalidResponse);
        }
        Ok(blocks)
    }

    /// Get all confirmed [`Utxo`]s locked to the given [`Address`].
    pub async fn get_address_utxos(&self, address: &Address) -> Result<Vec<Utxo>, Error> {
        let path = format!("/address/{address}/utxo");

        self.get_response_json(&path).await
    }

    /// Get all confirmed [`Utxo`]s locked to the given [`Script`].
    pub async fn get_scripthash_utxos(&self, script: &Script) -> Result<Vec<Utxo>, Error> {
        let script_hash = sha256::Hash::hash(script.as_bytes());
        let path = format!("/scripthash/{script_hash}/utxo");

        self.get_response_json(&path).await
    }
}

/// Truncate a request path to its first segment, which identifies the endpoint
/// without the user-specific parameters that follow.
///
/// ```text
/// path  "/tx/abc123/raw"
///   =>  "/tx"
/// ```
fn first_path_segment(path: &str) -> &str {
    // Skip the leading '/', then truncate at the next one.
    match path.get(1..).and_then(|rest| rest.find('/')) {
        Some(idx) => &path[..idx + 1],
        None => path,
    }
}

/// A trait for abstracting over async sleep implementations.
///
/// [`AsyncClient`] uses this trait to wait between retry attempts without
/// committing the client type to a specific async runtime.
///
/// The only provided implementation is [`DefaultSleeper`], which is backed by Tokio.
/// Custom implementations can be provided to support other runtimes.
pub trait Sleeper: 'static {
    /// The [`Future`](std::future::Future) type returned by [`Sleeper::sleep`].
    type Sleep: std::future::Future<Output = ()>;
    /// Return a [`Future`](std::future::Future) that completes after `duration`.
    fn sleep(dur: Duration) -> Self::Sleep;
}

/// The default [`Sleeper`] implementation, backed by [`tokio::time::sleep`].
///
/// This type is available when the `tokio` feature is enabled or while running
/// tests.
#[derive(Debug, Clone, Copy)]
pub struct DefaultSleeper;

#[cfg(any(test, feature = "tokio"))]
impl Sleeper for DefaultSleeper {
    type Sleep = tokio::time::Sleep;

    fn sleep(dur: std::time::Duration) -> Self::Sleep {
        tokio::time::sleep(dur)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// $ cargo test -p esplora-client --lib -- test_first_path_segment
    #[test]
    fn test_first_path_segment() {
        #[track_caller]
        fn test(path: &str, expected: &str) {
            assert_eq!(first_path_segment(path), expected, "path: {path}");
        }

        test("", "");
        test("/", "/");
        test("/tx", "/tx");
        test("/tx/abc123", "/tx");
        test("/tx/abc123/raw", "/tx");
        test("/scripthash/abc123/txs", "/scripthash");
    }
}
