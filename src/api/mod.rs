//! HTTP API: `/v1/systemone` (drop-in for qev), `/v1/systemone:batch` (extension), `/health`, `/v1/models`,
//! `/metrics` (DESIGN.md §7). Owner: W5.
//!
//! - [`server`]: the axum router and handlers, and the request flow: body budget → CPU pool (parse, validate,
//!   plan) → scheduler → CPU pool (assemble) → response. Its module docs list every client-visible difference
//!   from qev.
//! - [`schema`]: request and response types, with validation that reproduces pydantic's 422 bodies exactly.
//! - [`query`]: query-string parsing with Starlette/Python semantics (`debug`, `debias`).
//! - [`client`]: a small HTTP/1.1 client for tests and `loadgen`.
//!
//! See docs/ARCHITECTURE.md §2 (request lifecycle) and §4.2.

pub mod client;
pub mod query;
pub mod schema;
pub mod server;
