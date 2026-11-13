//! The checkpoint's `tokenizer.json` through HF tokenizers (the library behind transformers' fast tokenizer), with
//! `add_special_tokens=False` as qev's `_encode` uses it. Offsets are in Unicode code points, like Python's
//! `return_offsets_mapping=True`.

use anyhow::{Result, anyhow, bail, ensure};
use std::path::Path;

use crate::frontend::render::{LETTERS, slot_labels};

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Ok(Tokenizer { inner })
    }

    /// `tokenizer.encode(text, add_special_tokens=False)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Token ids and each token's `[start, end)` in code points of `text`.
    pub fn encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let enc = self.inner.encode_char_offsets(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("decoding: {e}"))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// qev's `_verify_slots`: the token id of every slot label that is one round-trip token. A–Z are required; the
    /// two-letter labels are all or nothing (a tokenizer that splits any keeps 26 slots).
    pub fn verify_slots(&self) -> Result<Vec<u32>> {
        let one = |label: &str| -> Result<Option<u32>> {
            let ids = self.encode(label)?;
            Ok(if ids.len() == 1 && self.decode(&ids)? == label { Some(ids[0]) } else { None })
        };
        let labels = slot_labels();
        let mut ids = Vec::with_capacity(labels.len());
        let mut bad = Vec::new();
        for l in &labels[..LETTERS.len()] {
            match one(l)? {
                Some(id) => ids.push(id),
                None => bad.push(l.as_str()),
            }
        }
        if !bad.is_empty() {
            bail!("answer slots {bad:?} are not single round-trip tokens");
        }
        let extra: Vec<Option<u32>> = labels[LETTERS.len()..].iter().map(|l| one(l)).collect::<Result<_>>()?;
        if extra.iter().all(Option::is_some) {
            ids.extend(extra.into_iter().flatten());
        }
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        ensure!(sorted.len() == ids.len(), "answer slot tokens collide");
        Ok(ids)
    }
}
