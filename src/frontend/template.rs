//! The checkpoint's chat template (`chat_template.jinja`), rendered with minijinja the way transformers'
//! `apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)` renders it with
//! Jinja2: `trim_blocks` and `lstrip_blocks` on, no autoescape, Python string methods available (`startswith`,
//! `split`, ...) and a `raise_exception` function.

use anyhow::{Context, Result, anyhow};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, context};
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
}

const NAME: &str = "chat_template.jinja";

impl ChatTemplate {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(NAME);
        let source = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_source(source)
    }

    pub fn from_source(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned(NAME, source).context("compiling the chat template")?;
        Ok(ChatTemplate { env })
    }

    /// The prompt for `[{"role": "system", "content": system}, {"role": "user", "content": user}]` with the
    /// generation prompt and thinking disabled (qev's `_render_prompt`).
    pub fn render(&self, system: &str, user: &str) -> Result<String> {
        let messages = vec![
            context! { role => "system", content => system },
            context! { role => "user", content => user },
        ];
        let tmpl = self.env.get_template(NAME)?;
        tmpl.render(context! { messages => messages, add_generation_prompt => true, enable_thinking => false })
            .map_err(|e| anyhow!("rendering the chat template: {e:#}"))
    }

    /// Render one branch prompt and cut it around `suffix` (located with `rfind`, as serve.patch's `_split` does
    /// with `rindex`): returns `(head, tail)` with `prompt == head + suffix + tail`.
    ///
    /// For qev's prompts the template is linear in the user turn: the turn always starts with `<state>` and ends
    /// with the answer line's `.`, so the template's `|trim` never touches it, and every branch of a request
    /// shares this head and tail. The frontend renders the template once per request through this.
    pub fn frame(&self, system: &str, state_block: &str, suffix: &str) -> Result<(String, String)> {
        let mut user = String::with_capacity(state_block.len() + suffix.len());
        user.push_str(state_block);
        user.push_str(suffix);
        let mut prompt = self.render(system, &user)?;
        let at = prompt.rfind(suffix).context("rendered prompt does not contain the question suffix")?;
        let tail = prompt[at + suffix.len()..].to_string();
        prompt.truncate(at);
        Ok((prompt, tail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "/home/velvet/Desktop/hf_models/elephant-1";

    #[test]
    fn renders_like_transformers() {
        let Ok(t) = ChatTemplate::load(Path::new(MODEL)) else {
            eprintln!("skipped: no checkpoint at {MODEL}");
            return;
        };
        let p = t.render("Sys.", "<state>\nx\n</state>\n\nQ.").unwrap();
        assert_eq!(
            p,
            "<|im_start|>system\nSys.<|im_end|>\n<|im_start|>user\n<state>\nx\n</state>\n\nQ.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let (h, tail) = t.frame("Sys.", "<state>\nQ.\n</state>\n\n", "Q.").unwrap();
        assert!(h.ends_with("<state>\nQ.\n</state>\n\n"));
        assert_eq!(tail, "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
