//! HTTP API: `/v1/systemone` (drop-in for qev), `/health`, `/v1/models`, `/metrics` (DESIGN.md §7). Owner: W5.

pub mod client;
pub mod query;
pub mod schema;
pub mod server;
