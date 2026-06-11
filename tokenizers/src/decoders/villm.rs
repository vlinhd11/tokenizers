use serde::{Deserialize, Serialize};

use crate::tokenizer::Result;

/// ViLLM decoder.
///
/// Handles:
/// - Byte tokens (`<0xNN>`) merged into decoded UTF-8 sequences
/// - Code-switch markers removed from output
/// - Smart spacing using language-region state machine
///   - Vi region: each token is a complete word → add space
///   - EN region: tokens with `▁` prefix start new words; continuations fuse
/// - `_` and `\u{2581}` replaced with spaces
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ViLLMDecoder {
    cs_vi_en: String,
    cs_en_vi: String,
}

impl ViLLMDecoder {
    pub fn new(cs_vi_en: String, cs_en_vi: String) -> Self {
        Self {
            cs_vi_en,
            cs_en_vi,
        }
    }

    /// Merge byte tokens into adjacent text, keep CS markers and `▁` intact.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for token in tokens {
            if token == self.cs_vi_en || token == self.cs_en_vi {
                if !byte_buf.is_empty() {
                    out.push(Self::flush_bytes(&mut byte_buf));
                }
                out.push(token.clone());
                continue;
            }

            if token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
            {
                if let Ok(b) = u8::from_str_radix(&token[3..5], 16) {
                    byte_buf.push(b);
                    continue;
                }
            }

            // Non-byte, non-CS token: flush pending bytes into it
            if !byte_buf.is_empty() {
                let decoded = Self::flush_bytes(&mut byte_buf);
                out.push(decoded + &token);
            } else {
                out.push(token);
            }
        }
        // Trailing bytes → append to last token or create new
        if !byte_buf.is_empty() {
            let decoded = Self::flush_bytes(&mut byte_buf);
            if let Some(last) = out.last_mut() {
                last.push_str(&decoded);
            } else {
                out.push(decoded);
            }
        }
        Ok(out)
    }

    /// Decode with language-region-aware spacing.
    ///
    /// - Vi mode (no `▁` seen): each token is a complete word → add smart space
    /// - EN mode (`▁` seen): `▁`-prefixed tokens start new words; others are continuations (fuse)
    /// - CS markers are skipped
    /// - `_` within tokens replaced with space (Vi compounds)
    pub fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let parts = self.decode_chain(tokens)?;
        let mut result = String::new();
        let mut en_mode = false;

        for p in &parts {
            // CS markers toggle language mode
            if *p == self.cs_vi_en {
                en_mode = true;
                continue;
            }
            if *p == self.cs_en_vi {
                en_mode = false;
                continue;
            }

            let has_bs = p.starts_with('\u{2581}');
            let cleaned = p.replace('\u{2581}', "").replace('_', " ");

            if result.is_empty() {
                result.push_str(cleaned.trim_start());
                if has_bs {
                    en_mode = true;
                }
                continue;
            }

            if en_mode {
                if has_bs {
                    // word boundary
                    if !cleaned.is_empty() {
                        Self::smart_append(&mut result, &cleaned);
                    }
                } else {
                    // continuation: no space
                    result.push_str(&cleaned);
                }
            } else if has_bs {
                // first ▁ switches to EN mode
                en_mode = true;
                if !cleaned.is_empty() {
                    Self::smart_append(&mut result, &cleaned);
                }
            } else if !cleaned.is_empty() {
                // Vi token: each is a separate word
                Self::smart_append(&mut result, &cleaned);
            }
        }
        Ok(result)
    }

    /// Append `s` to `result` with smart spacing (space unless punctuation or bracket rule).
    fn smart_append(result: &mut String, s: &str) {
        if let Some(c) = s.chars().next() {
            let last = result.chars().next_back().unwrap();
            if matches!(
                c,
                '.' | ',' | '!' | '?' | ';' | ':' | '%'
                    | ')' | ']' | '}' | '"' | '\''
            ) || matches!(last, '(' | '[' | '{')
            {
                result.push_str(s);
            } else {
                result.push(' ');
                result.push_str(s);
            }
        }
    }
}

impl ViLLMDecoder {
    fn flush_bytes(buf: &mut Vec<u8>) -> String {
        let s = String::from_utf8_lossy(buf).to_string();
        buf.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_decode() {
        let decoder = ViLLMDecoder::new("<cs_vi_en>".into(), "<cs_en_vi>".into());

        // Single byte token → character
        let res = decoder.decode_chain(vec!["<0x61>".into()]).unwrap();
        assert_eq!(res.join(""), "a");

        // Multi-byte UTF-8 sequence (叫 = E5 8F AB)
        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8F>".into(), "<0xAB>".into()])
            .unwrap();
        assert_eq!(res.join(""), "叫");

        // Mixed byte and regular tokens
        let res = decoder
            .decode_chain(vec![
                "<0xE5>".into(),
                "<0x8F>".into(),
                "<0xAB>".into(),
                "hello".into(),
            ])
            .unwrap();
        assert_eq!(res.join(""), "叫hello");

        // Code-switch markers removed in decode()
        let res = decoder
            .decode(vec!["hello".into(), "<cs_vi_en>".into(), "\u{2581}world".into()])
            .unwrap();
        assert_eq!(res, "hello world");

        // Byte fusion with adjacent token
        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8F>".into(), "<0xAB>".into(), "hello".into()])
            .unwrap();
        assert_eq!(res.join(""), "叫hello");
    }

    #[test]
    fn test_smart_spacing_vi() {
        let decoder = ViLLMDecoder::new("<cs_vi_en>".into(), "<cs_en_vi>".into());

        // Vi tokens without ▁: each is a separate word
        let res = decoder.decode(vec!["Xin".into(), "chào".into(), "thế_giới".into()]).unwrap();
        assert_eq!(res, "Xin chào thế giới");
    }

    #[test]
    fn test_smart_spacing_en() {
        let decoder = ViLLMDecoder::new("<cs_vi_en>".into(), "<cs_en_vi>".into());

        // EN tokens with ▁: word starts
        let res = decoder.decode(vec!["\u{2581}machine".into(), "\u{2581}learning".into()]).unwrap();
        assert_eq!(res, "machine learning");
    }

    #[test]
    fn test_smart_spacing_en_continuation() {
        let decoder = ViLLMDecoder::new("<cs_vi_en>".into(), "<cs_en_vi>".into());

        // EN tokens with ▁ on first token, continuations without ▁ fuse together
        let res = decoder
            .decode(vec![
                "\u{2581}abc".into(),
                "X".into(),
                "Y".into(),
                "Z".into(),
                "123".into(),
            ])
            .unwrap();
        assert_eq!(res, "abcXYZ123");
    }

    #[test]
    fn test_smart_spacing_punct() {
        let decoder = ViLLMDecoder::new("<cs_vi_en>".into(), "<cs_en_vi>".into());

        // Space before punctuation removed, but space after retained
        let res = decoder.decode(vec!["hello".into(), ",".into(), "world".into()]).unwrap();
        assert_eq!(res, "hello, world");

        // No extra space for opening brackets
        let res = decoder.decode(vec!["(".into(), "hello".into(), ")".into()]).unwrap();
        assert_eq!(res, "(hello)");

        // Normal space between words
        let res = decoder.decode(vec!["hello".into(), "world".into()]).unwrap();
        assert_eq!(res, "hello world");
    }

    #[test]
    fn test_smart_spacing_mixed() {
        let decoder = ViLLMDecoder::new("[VI→EN]".into(), "[EN→VI]".into());

        // Mixed Vi/EN with CS markers
        let res = decoder
            .decode(vec![
                "Tôi_muốn".into(),
                "học".into(),
                "[VI→EN]".into(),
                "\u{2581}machine".into(),
                "\u{2581}learning".into(),
                "[EN→VI]".into(),
                "và".into(),
                "[VI→EN]".into(),
                "\u{2581}deep".into(),
                "\u{2581}learning".into(),
            ])
            .unwrap();
        assert_eq!(res, "Tôi muốn học machine learning và deep learning");
    }
}
