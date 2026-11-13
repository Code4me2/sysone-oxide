//! A minimal keep-alive HTTP/1.1 client over one connection, for the end-to-end tests and the load generator.

use anyhow::{Context, Result};
use axum::body::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1::{SendRequest, handshake};
use hyper::{HeaderMap, Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;

pub struct Client {
    authority: String,
    sender: SendRequest<Full<Bytes>>,
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Reply {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

impl Client {
    /// Connect to `host:port`.
    pub async fn connect(addr: &str) -> Result<Self> {
        let stream = TcpStream::connect(addr).await.with_context(|| format!("connecting to {addr}"))?;
        stream.set_nodelay(true)?;
        let (sender, conn) = handshake(TokioIo::new(stream)).await?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        Ok(Client { authority: addr.to_string(), sender })
    }

    pub async fn request(&mut self, method: Method, path: &str, body: impl Into<Bytes>) -> Result<Reply> {
        self.sender.ready().await?;
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("host", &self.authority)
            .header("content-type", "application/json")
            .body(Full::new(body.into()))?;
        let resp = self.sender.send_request(req).await?;
        let (parts, body) = resp.into_parts();
        let body = body.collect().await?.to_bytes();
        Ok(Reply { status: parts.status, headers: parts.headers, body })
    }

    pub async fn post(&mut self, path: &str, body: impl Into<Bytes>) -> Result<Reply> {
        self.request(Method::POST, path, body).await
    }

    pub async fn get(&mut self, path: &str) -> Result<Reply> {
        self.request(Method::GET, path, Bytes::new()).await
    }
}
