use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::tokenizer::{Model, Result, Token};
use regex::Regex;

// ── Language Detection ─────────────────────────────────────────

fn build_vi_chars() -> HashSet<char> {
    "àáảãạăắặằẳẵâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵÀÁẢÃẠĂẮẶẰẲẴÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
        .chars()
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lang {
    Vi,
    En,
    Num,
    Code,
    Punct,
}

fn is_numeric(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(|c| {
            c.is_ascii_digit() || matches!(c, '.' | ',' | '/' | '%' | '$' | '+' | '-')
        })
}

// ── SP trie for English subword ─────────────────────────────────

#[derive(Clone, Debug)]
struct SpTrieNode {
    children: HashMap<char, SpTrieNode>,
    score: f64,
    is_end: bool,
}

impl Default for SpTrieNode {
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            score: 0.0,
            is_end: false,
        }
    }
}

impl SpTrieNode {
    fn insert(&mut self, piece: &str, score: f64) {
        let mut node = self;
        for c in piece.chars() {
            node = node.children.entry(c).or_default();
        }
        node.score = score;
        node.is_end = true;
    }

    fn matches<'a>(&'a self, text: &'a str, start: usize) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        let mut node = self;
        for (i, c) in text[start..].char_indices() {
            if let Some(child) = node.children.get(&c) {
                node = child;
                if node.is_end {
                    results.push((start + i + c.len_utf8(), node.score));
                }
            } else {
                break;
            }
        }
        results
    }
}

const STRIP_CHARS: &[char] = &[
    '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', '\'',
];

// ── Serialization helper ───────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct ViLLMConfig {
    #[serde(rename = "type")]
    #[serde(default = "default_villm_type")]
    type_: String,
    token2id: HashMap<String, u32>,
    unk_token: String,
    vi_syllables: Vec<String>,
    base_forms: Vec<String>,
    vi_compounds: HashMap<String, f64>,
    compound_unigram_score: f64,
    sp_pieces: Vec<(String, f64)>,
    cs_vi_en: String,
    cs_en_vi: String,
}

fn default_villm_type() -> String {
    "ViLLM".to_string()
}

// ── Core model ─────────────────────────────────────────────────

impl PartialEq for ViLLMModel {
    fn eq(&self, other: &Self) -> bool {
        self.token2id == other.token2id
            && self.unk_token == other.unk_token
            && self.unk_id == other.unk_id
            && self.vi_compounds == other.vi_compounds
            && self.compound_unigram_score == other.compound_unigram_score
            && self.re_code.as_str() == other.re_code.as_str()
            && self.cs_vi_en == other.cs_vi_en
            && self.cs_en_vi == other.cs_en_vi
            && self.vi_syllables == other.vi_syllables
            && self.base_forms == other.base_forms
            && self.sp_vocab == other.sp_vocab
            && self.byte_ids == other.byte_ids
    }
}

#[derive(Clone, Debug)]
pub struct ViLLMModel {
    token2id: HashMap<String, u32>,
    id2token: HashMap<u32, String>,
    unk_token: String,
    unk_id: u32,
    vi_chars: HashSet<char>,
    vi_syllables: HashSet<String>,
    base_forms: HashSet<String>,
    vi_compounds: HashMap<String, f64>,
    compound_unigram_score: f64,
    sp_trie: SpTrieNode,
    sp_vocab: HashMap<String, u32>,
    byte_ids: HashMap<u8, u32>,
    re_code: Regex,
    cs_vi_en: String,
    cs_en_vi: String,
}

impl ViLLMModel {
    fn from_config(cfg: ViLLMConfig) -> Self {
        let id2token: HashMap<u32, String> =
            cfg.token2id.iter().map(|(k, v)| (*v, k.clone())).collect();
        let unk_id = cfg.token2id.get(&cfg.unk_token).copied().unwrap_or(1);
        let re_code = Regex::new(r"[a-z][A-Z]|[A-Z]{2,}[a-z]").unwrap();

        let mut byte_ids = HashMap::new();
        for b in 0..=255u8 {
            let tok = format!("<0x{b:02X}>");
            if let Some(&id) = cfg.token2id.get(&tok) {
                byte_ids.insert(b, id);
            }
        }

        let mut sp_trie = SpTrieNode::default();
        let mut sp_vocab = HashMap::new();
        for (piece, score) in &cfg.sp_pieces {
            sp_trie.insert(piece, *score);
            if let Some(&id) = cfg.token2id.get(piece) {
                sp_vocab.insert(piece.clone(), id);
            }
        }

        Self {
            token2id: cfg.token2id,
            id2token,
            unk_token: cfg.unk_token,
            unk_id,
            vi_chars: build_vi_chars(),
            vi_syllables: cfg.vi_syllables.into_iter().collect(),
            base_forms: cfg.base_forms.into_iter().collect(),
            vi_compounds: cfg.vi_compounds,
            compound_unigram_score: cfg.compound_unigram_score,
            sp_trie,
            sp_vocab,
            byte_ids,
            re_code,
            cs_vi_en: cfg.cs_vi_en,
            cs_en_vi: cfg.cs_en_vi,
        }
    }

    // ── Language detection ──────────────────────────────────

    fn is_punct_token(&self, token: &str) -> bool {
        token.is_empty()
            || (token.chars().all(|c| !c.is_alphanumeric())
                && token.chars().any(|c| !c.is_whitespace()))
    }

    fn detect_lang(&self, token: &str) -> Lang {
        if self.is_punct_token(token) {
            return Lang::Punct;
        }
        if is_numeric(token) {
            return Lang::Num;
        }
        if self.re_code.is_match(token) {
            return Lang::Code;
        }
        let mut alpha_count = 0usize;
        let mut vi_count = 0usize;
        for c in token.chars() {
            if c.is_alphabetic() {
                alpha_count += 1;
                if self.vi_chars.contains(&c) {
                    vi_count += 1;
                }
            }
        }
        if alpha_count == 0 {
            return Lang::Num;
        }
        if (vi_count as f64) / (alpha_count as f64) >= 0.15 {
            return Lang::Vi;
        }
        if alpha_count == 1 && vi_count == 0 && token.is_ascii() {
            return Lang::En;
        }
        let lower = token.trim().to_lowercase();
        let stripped = lower.trim_end_matches(|c: char| ".,!?;:)".contains(c));
        if self.vi_syllables.contains(stripped)
            || self.base_forms.contains(stripped)
        {
            return Lang::Vi;
        }
        Lang::En
    }

    // ── SP subword tokenization ─────────────────────────────

    fn tokenize_sp(&self, word: &str) -> Vec<(String, u32)> {
        let mut input = String::with_capacity(word.len() + 4);
        input.push('\u{2581}');
        input.push_str(word);
        let n = input.len();

        if self.sp_vocab.contains_key(&input) {
            let id = self.sp_vocab[&input];
            return vec![(input.clone(), id)];
        }

        let mut dp = vec![f64::NEG_INFINITY; n + 1];
        let mut back = vec![0usize; n + 1];
        dp[0] = 0.0;

        for start in 0..n {
            if dp[start] == f64::NEG_INFINITY {
                continue;
            }
            for (end, score) in self.sp_trie.matches(&input, start) {
                let new_score = dp[start] + score;
                if new_score > dp[end] {
                    dp[end] = new_score;
                    back[end] = start;
                }
            }
        }

        if dp[n] == f64::NEG_INFINITY {
            return vec![(self.unk_token.clone(), self.unk_id)];
        }

        let mut tokens: Vec<(String, u32)> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let start = back[pos];
            let piece = &input[start..pos];
            let id = self.sp_vocab.get(piece).copied().unwrap_or(self.unk_id);
            tokens.push((piece.to_string(), id));
            pos = start;
        }
        tokens.reverse();
        tokens
    }

    fn tokenize_en_word(&self, word: &str) -> Vec<(String, u32)> {
        // Try exact match first (preserves case and punctuation)
        if let Some(&id) = self.token2id.get(word) {
            return vec![(word.to_string(), id)];
        }
        // For non-lowercase words: byte fallback preserves case
        if !word.chars().all(|c| c.is_lowercase()) {
            return self.byte_fallback(word);
        }
        // Try lowercase of exact word
        let lower = word.to_lowercase();
        if let Some(&id) = self.token2id.get(&lower) {
            return vec![(word.to_string(), id)];
        }
        // Try stripping trailing punctuation, then match
        let stripped_tail = word.trim_end_matches(STRIP_CHARS);
        if stripped_tail.len() < word.len() && !stripped_tail.is_empty() {
            if let Some(&id) = self.token2id.get(stripped_tail) {
                return vec![(word.to_string(), id)];
            }
            let lower_stripped = stripped_tail.to_lowercase();
            if let Some(&id) = self.token2id.get(&lower_stripped) {
                return vec![(word.to_string(), id)];
            }
        }
        // Try SP subword on lowercased version (SP trie is case-sensitive, lowercase only)
        let sp_tokens = self.tokenize_sp(&lower);
        if !sp_tokens.is_empty()
            && sp_tokens.iter().any(|(t, _)| t != &self.unk_token)
        {
            return sp_tokens;
        }
        // Fallback: byte-fallback for characters not in vocab
        self.byte_fallback(word)
    }

    fn byte_fallback(&self, text: &str) -> Vec<(String, u32)> {
        text.bytes()
            .map(|b| {
                let id = self.byte_ids.get(&b).copied().unwrap_or(self.unk_id);
                (format!("<0x{b:02X}>"), id)
            })
            .collect()
    }

    // ── VI tokenization ─────────────────────────────────────

    fn try_split_compound(&self, word: &str) -> Vec<(String, u32)> {
        // Use original word for matching; lowercase only for lookup
        // Use char_indices to avoid splitting in the middle of multi-byte chars
        let chars: Vec<(usize, char)> = word.char_indices().collect();
        let n = chars.len();
        for i in 1..n {
            let split_byte = chars[i].0;
            let left = &word[..split_byte];
            let right = &word[split_byte..];
            if left.len() >= 2
                && right.len() >= 2
                && self.vi_syllables.contains(&left.to_lowercase())
                && self.vi_syllables.contains(&right.to_lowercase())
            {
                let mut out = Vec::with_capacity(2);
                // Try exact match first for case preservation
                if let Some(&id) = self.token2id.get(left) {
                    out.push((left.to_string(), id));
                } else if let Some(&id) = self.token2id.get(&left.to_lowercase()) {
                    out.push((left.to_string(), id));
                }
                if let Some(&id) = self.token2id.get(right) {
                    out.push((right.to_string(), id));
                } else if let Some(&id) = self.token2id.get(&right.to_lowercase()) {
                    out.push((right.to_string(), id));
                }
                if out.len() == 2 {
                    return out;
                }
            }
        }
        vec![]
    }

    fn compound_key(&self, a: &str, b: &str) -> String {
        let mut key = String::with_capacity(a.len() + b.len() + 1);
        for c in a.chars() {
            key.extend(c.to_lowercase());
        }
        key.push('_');
        for c in b.chars() {
            key.extend(c.to_lowercase());
        }
        key
    }

    fn tokenize_vi_word(&self, word: &str) -> Vec<(String, u32)> {
        // Preserve original form — try exact match first
        if let Some(&id) = self.token2id.get(word) {
            return vec![(word.to_string(), id)];
        }
        let lower = word.to_lowercase();
        if let Some(&id) = self.token2id.get(&lower) {
            return vec![(word.to_string(), id)];
        }
        // Strip trailing punctuation, try match
        let stripped = word.trim_end_matches(STRIP_CHARS);
        if stripped.len() < word.len() && !stripped.is_empty() {
            if let Some(&id) = self.token2id.get(stripped) {
                return vec![(word.to_string(), id)];
            }
            let lower_stripped = stripped.to_lowercase();
            if let Some(&id) = self.token2id.get(&lower_stripped) {
                return vec![(word.to_string(), id)];
            }
        }
        let split = self.try_split_compound(word);
        if !split.is_empty() {
            return split;
        }
        self.byte_fallback(word)
    }

    fn cased_compound(&self, compound: &str, orig_words: &[&str]) -> String {
        let parts: Vec<&str> = compound.split('_').collect();
        let mut cased = Vec::with_capacity(parts.len());
        for (i, &part) in parts.iter().enumerate() {
            if i < orig_words.len() {
                let o = orig_words[i];
                if o.chars().all(|c| c.is_uppercase()) {
                    cased.push(part.to_uppercase());
                } else if o.chars().next().map_or(false, |c| c.is_uppercase()) {
                    let mut s = part.to_string();
                    if let Some(first) = s.chars().next() {
                        s.replace_range(0..first.len_utf8(), &first.to_uppercase().to_string());
                    }
                    cased.push(s);
                } else {
                    cased.push(part.to_string());
                }
            } else {
                cased.push(part.to_string());
            }
        }
        cased.join("_")
    }

    // ── Pre-tokenization ────────────────────────────────────

    fn is_combining_mark(c: char) -> bool {
        let code = c as u32;
        (code >= 0x0300 && code <= 0x036F)
            || (code >= 0x0483 && code <= 0x0489)
            || (code >= 0x0591 && code <= 0x05BD)
            || (code >= 0x0610 && code <= 0x061A)
            || (code >= 0x064B && code <= 0x065F)
            || (code >= 0x0670) && (code <= 0x06D6)
            || (code >= 0x0901 && code <= 0x0903)
            || (code >= 0x093A && code <= 0x093C)
            || (code >= 0x093E && code <= 0x094F)
            || (code >= 0x0951 && code <= 0x0957)
            || (code >= 0x0962 && code <= 0x0963)
            || (code >= 0x0981 && code <= 0x0983)
            || (code >= 0x09BC) && (code <= 0x09C4)
            || (code >= 0x09E2 && code <= 0x09E3)
            || (code >= 0x0A01 && code <= 0x0A03)
            || (code >= 0x0A3C) && (code <= 0x0A4F)
            || (code >= 0x0A81 && code <= 0x0A83)
            || (code >= 0x0ABC && code <= 0x0AC5)
            || (code >= 0x0AC7 && code <= 0x0AC9)
            || (code >= 0x0ACB && code <= 0x0ACD)
            || (code >= 0x0AE2 && code <= 0x0AE3)
            || (code >= 0x0B01 && code <= 0x0B03)
            || (code >= 0x0B3C && code <= 0x0B44)
            || (code >= 0x0B47 && code <= 0x0B48)
            || (code >= 0x0B4B && code <= 0x0B4D)
            || (code >= 0x0B56 && code <= 0x0B57)
            || (code >= 0x0B62 && code <= 0x0B63)
            || (code >= 0x0B82) && (code <= 0x0B83)
            || (code >= 0x0BC0 && code <= 0x0BCD)
            || (code >= 0x0C00 && code <= 0x0C03)
            || (code >= 0x0C3E && code <= 0x0C44)
            || (code >= 0x0C46 && code <= 0x0C48)
            || (code >= 0x0C4A && code <= 0x0C4D)
            || (code >= 0x0C55 && code <= 0x0C56)
            || (code >= 0x0C62 && code <= 0x0C63)
            || (code >= 0x0C81 && code <= 0x0C83)
            || (code >= 0x0CBC && code <= 0x0CC4)
            || (code >= 0x0CC6 && code <= 0x0CC8)
            || (code >= 0x0CCA && code <= 0x0CCD)
            || (code >= 0x0CD5 && code <= 0x0CD6)
            || (code >= 0x0CE2 && code <= 0x0CE3)
            || (code >= 0x0D01 && code <= 0x0D03)
            || (code >= 0x0D3E && code <= 0x0D44)
            || (code >= 0x0D46 && code <= 0x0D48)
            || (code >= 0x0D4A && code <= 0x0D4D)
            || (code >= 0x0D62 && code <= 0x0D63)
            || (code >= 0x0DCA) && (code <= 0x0DDF)
            || (code >= 0x0E31) && (code <= 0x0E3A)
            || (code >= 0x0E47 && code <= 0x0E4E)
            || (code >= 0x0EB1 && code <= 0x0EBB)
            || (code >= 0x0EC8 && code <= 0x0ECD)
            || (code >= 0x0F18 && code <= 0x0F19)
            || (code >= 0x0F35) && (code <= 0x0F39)
            || (code >= 0x0F3E && code <= 0x0F3F)
            || (code >= 0x0F71 && code <= 0x0F84)
            || (code >= 0x0F86 && code <= 0x0F87)
            || (code >= 0x0F8D && code <= 0x0FBC)
            || (code >= 0x0FC6)
            || (code >= 0x102B && code <= 0x103E)
            || (code >= 0x1056 && code <= 0x1059)
            || (code >= 0x1712 && code <= 0x1714)
            || (code >= 0x1732 && code <= 0x1734)
            || (code >= 0x1752 && code <= 0x1753)
            || (code >= 0x1772 && code <= 0x1773)
            || (code >= 0x17B4 && code <= 0x17D3)
            || (code >= 0x17DD)
            || (code >= 0x180B && code <= 0x180D)
            || (code >= 0x18A9)
            || (code >= 0x1920 && code <= 0x1938)
            || (code >= 0x1A17 && code <= 0x1A1B)
            || (code >= 0x1B00 && code <= 0x1B04)
            || (code >= 0x1B34 && code <= 0x1B44)
            || (code >= 0x1B6B && code <= 0x1B73)
            || (code >= 0x1B80 && code <= 0x1B82)
            || (code >= 0x1BA1 && code <= 0x1BAD)
            || (code >= 0x1BE6 && code <= 0x1BF3)
            || (code >= 0x1C24 && code <= 0x1C37)
            || (code >= 0x1CD0 && code <= 0x1CD2)
            || (code >= 0x1CD4 && code <= 0x1CE8)
            || (code >= 0x1CED)
            || (code >= 0x1CF2 && code <= 0x1CF4)
            || (code >= 0x1DC0 && code <= 0x1DFF)
            || (code >= 0x200C && code <= 0x200D)
            || (code >= 0x20D0 && code <= 0x20FF)
            || (code >= 0x2CEF && code <= 0x2CF1)
            || (code >= 0x2D7F)
            || (code >= 0x2DE0 && code <= 0x2DFF)
            || (code >= 0xA66F && code <= 0xA672)
            || (code >= 0xA674 && code <= 0xA67D)
            || (code >= 0xA69E && code <= 0xA69F)
            || (code >= 0xA802)
            || (code >= 0xA806)
            || (code >= 0xA80B)
            || (code >= 0xA823 && code <= 0xA827)
            || (code >= 0xA880 && code <= 0xA881)
            || (code >= 0xA8B4 && code <= 0xA8C4)
            || (code >= 0xA8E0 && code <= 0xA8F1)
            || (code >= 0xA926 && code <= 0xA92D)
            || (code >= 0xA947 && code <= 0xA953)
            || (code >= 0xA980 && code <= 0xA983)
            || (code >= 0xA9B3 && code <= 0xA9C0)
            || (code >= 0xAAB0)
            || (code >= 0xAAB2 && code <= 0xAAB4)
            || (code >= 0xAAB7 && code <= 0xAAB8)
            || (code >= 0xAABE && code <= 0xAABF)
            || (code >= 0xAAC1)
            || (code >= 0xAAEB && code <= 0xAAEF)
            || (code >= 0xAAF5)
            || (code >= 0xABE3 && code <= 0xABEA)
            || (code >= 0xFB1E)
            || (code >= 0xFE00 && code <= 0xFE0F)
            || (code >= 0xFE20 && code <= 0xFE2F)
            || (code >= 0x101FD)
            || (code >= 0x10A01 && code <= 0x10A03)
            || (code >= 0x10A05 && code <= 0x10A06)
            || (code >= 0x10A0C && code <= 0x10A0F)
            || (code >= 0x10A38 && code <= 0x10A3A)
            || (code >= 0x10A3F)
            || (code >= 0x11000 && code <= 0x11002)
            || (code >= 0x11038 && code <= 0x11046)
            || (code >= 0x11080 && code <= 0x11082)
            || (code >= 0x110B0 && code <= 0x110BA)
            || (code >= 0x11100 && code <= 0x11102)
            || (code >= 0x11127 && code <= 0x11134)
            || (code >= 0x11180 && code <= 0x11182)
            || (code >= 0x111B3 && code <= 0x111C0)
            || (code >= 0x1122C && code <= 0x11237)
            || (code >= 0x1123E)
            || (code >= 0x11301 && code <= 0x11303)
            || (code >= 0x1133E && code <= 0x11344)
            || (code >= 0x11347 && code <= 0x11348)
            || (code >= 0x1134B && code <= 0x1134D)
            || (code >= 0x11362 && code <= 0x11363)
            || (code >= 0x11366 && code <= 0x1136C)
            || (code >= 0x114B0 && code <= 0x114C3)
            || (code >= 0x115AF && code <= 0x115B5)
            || (code >= 0x115B8 && code <= 0x115C0)
            || (code >= 0x11630 && code <= 0x11640)
            || (code >= 0x116AB && code <= 0x116B7)
            || (code >= 0x1D165 && code <= 0x1D169)
            || (code >= 0x1D16D && code <= 0x1D172)
            || (code >= 0x1D17B && code <= 0x1D182)
            || (code >= 0x1D185 && code <= 0x1D18B)
            || (code >= 0x1D1AA && code <= 0x1D1AD)
            || (code >= 0x1D242 && code <= 0x1D244)
            || (code >= 0xE0100 && code <= 0xE01EF)
    }

    fn pretokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut current_type: Option<&str> = None;

        for c in text.chars() {
            if c.is_whitespace() {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                // Preserve whitespace as single-char tokens
                result.push(c.to_string());
                current_type = None;
                continue;
            }
            // Combining marks inherit the type of the preceding character
            if Self::is_combining_mark(c) && current_type.is_some() {
                current.push(c);
                continue;
            }
            let is_alpha = c.is_alphanumeric();
            let tok_type = if is_alpha { "alpha" } else { "non_alpha" };
            match current_type {
                None => {
                    current.push(c);
                    current_type = Some(tok_type);
                }
                Some(t) => {
                    if tok_type == t {
                        current.push(c);
                    } else {
                        result.push(std::mem::take(&mut current));
                        current.push(c);
                        current_type = Some(tok_type);
                    }
                }
            }
        }
        if !current.is_empty() {
            result.push(current);
        }
        Self::fuse_operators(&mut result);
        result
    }

    fn fuse_operators(tokens: &mut Vec<String>) {
        let mut i = 0;
        while i + 1 < tokens.len() {
            let pair = format!("{}{}", tokens[i], tokens[i + 1]);
            let fused = matches!(
                pair.as_str(),
                "->"
                    | "=>"
                    | "!="
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "%="
                    | ">="
                    | "<="
                    | "::"
                    | ".."
                    | "**"
                    | "//"
            );
            if fused {
                tokens[i] = pair;
                tokens.remove(i + 1);
            }
            i += 1;
        }
    }

    // ── Code-switch markers ─────────────────────────────────

    fn maybe_add_cs(
        &self,
        result: &mut Vec<Token>,
        prev: Option<Lang>,
        curr: Lang,
        offset: (usize, usize),
    ) {
        match (prev, curr) {
            (Some(Lang::Vi), Lang::En) => {
                if let Some(&id) = self.token2id.get(&self.cs_vi_en) {
                    result.push(Token::new(id, self.cs_vi_en.clone(), offset));
                }
            }
            (Some(Lang::En), Lang::Vi) => {
                if let Some(&id) = self.token2id.get(&self.cs_en_vi) {
                    result.push(Token::new(id, self.cs_en_vi.clone(), offset));
                }
            }
            _ => {}
        }
    }

    // ── VI run Viterbi ──────────────────────────────────────

    fn tokenize_vi_run(
        &self,
        run_lower: &[String],
        run_raw: &[&str],
        word_offsets: &[(usize, usize)],
        result: &mut Vec<Token>,
    ) {
        let rn = run_lower.len();
        if rn == 0 {
            return;
        }

        use std::f64::NEG_INFINITY;
        let mut dp_score = vec![NEG_INFINITY; rn + 1];
        let mut dp_back = vec![-1isize; rn + 1];
        dp_score[0] = 0.0;
        for k in 1..=rn {
            dp_score[k] = dp_score[k - 1] + self.compound_unigram_score;
            dp_back[k] = k as isize - 1;
            if k >= 2 {
                let compound = self.compound_key(&run_lower[k - 2], &run_lower[k - 1]);
                if let Some(c_score) = self.vi_compounds.get(&compound) {
                    let score = dp_score[k - 2] + c_score;
                    if score > dp_score[k] {
                        dp_score[k] = score;
                        dp_back[k] = k as isize - 2;
                    }
                }
            }
        }

        let mut segments: Vec<String> = Vec::new();
        let mut seg_orig: Vec<Vec<String>> = Vec::new();
        let mut seg_range: Vec<(usize, usize)> = Vec::new();
        let mut pos = rn;
        while pos > 0 {
            let prev = dp_back[pos] as usize;
            if pos - prev == 2 {
                segments.push(format!("{}_{}", run_lower[prev], run_lower[prev + 1]));
                seg_orig.push(vec![
                    run_raw[prev].to_string(),
                    run_raw[prev + 1].to_string(),
                ]);
                seg_range.push((word_offsets[prev].0, word_offsets[prev + 1].1));
            } else {
                segments.push(run_lower[prev].clone());
                seg_orig.push(vec![run_raw[prev].to_string()]);
                seg_range.push(word_offsets[prev]);
            }
            pos = prev;
        }
        segments.reverse();
        seg_orig.reverse();
        seg_range.reverse();

        for (seg_idx, seg) in segments.iter().enumerate() {
            let offset = seg_range[seg_idx];
            let orig_refs: Vec<&str> = seg_orig[seg_idx].iter().map(|s| s.as_str()).collect();
            if seg.contains('_') {
                let cased = self.cased_compound(seg, &orig_refs);
                let variants = [cased.as_str(), &seg.to_uppercase(), seg.as_str()];
                let mut found = false;
                for &v in &variants {
                    if let Some(&id) = self.token2id.get(v) {
                        result.push(Token::new(id, v.to_string(), offset));
                        found = true;
                        break;
                    }
                }
                if !found {
                    for part in seg.split('_') {
                        let sub_tokens = self.tokenize_en_word(part);
                        for (t, id) in &sub_tokens {
                            result.push(Token::new(*id, t.clone(), offset));
                        }
                    }
                }
            } else {
                let orig = orig_refs[0];
                let is_upper = orig.chars().next().map_or(false, |c| c.is_uppercase());
                let is_all_upper = orig.chars().all(|c| c.is_uppercase());
                if is_all_upper {
                    let variants = [seg.to_uppercase(), seg.to_string()];
                    let mut found = false;
                    for v in &variants {
                        if let Some(&id) = self.token2id.get(v.as_str()) {
                            result.push(Token::new(id, v.clone(), offset));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let sub_tokens = self.tokenize_en_word(seg);
                        for (t, id) in &sub_tokens {
                            result.push(Token::new(*id, t.clone(), offset));
                        }
                    }
                } else if is_upper {
                    let mut titled = seg.to_string();
                    if let Some(first) = titled.chars().next() {
                        titled
                            .replace_range(0..first.len_utf8(), &first.to_uppercase().to_string());
                    }
                    let variants = [titled, seg.to_uppercase(), seg.to_string()];
                    let mut found = false;
                    for v in &variants {
                        if let Some(&id) = self.token2id.get(v.as_str()) {
                            result.push(Token::new(id, v.clone(), offset));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let sub_tokens = self.tokenize_en_word(seg);
                        for (t, id) in &sub_tokens {
                            result.push(Token::new(*id, t.clone(), offset));
                        }
                    }
                } else if let Some(&id) = self.token2id.get(seg.as_str()) {
                    result.push(Token::new(id, orig.to_string(), offset));
                } else {
                    let sub_tokens = self.tokenize_en_word(seg);
                    for (t, id) in &sub_tokens {
                        result.push(Token::new(*id, t.clone(), offset));
                    }
                }
            }
        }
    }
}

impl Model for ViLLMModel {
    type Trainer = crate::models::TrainerWrapper;

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }

        let words = self.pretokenize(sequence);
        let n = words.len();

        // Compute word offsets in original text
        let mut word_offsets: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut cursor = 0usize;
        for w in &words {
            let start = sequence[cursor..]
                .find(w.as_str())
                .map(|p| cursor + p)
                .unwrap_or(cursor);
            let end = start + w.len();
            word_offsets.push((start, end));
            cursor = end;
        }

        let mut result: Vec<Token> = Vec::new();
        let mut prev_lang: Option<Lang> = None;

        let mut i = 0;
        while i < n {
            let word = &words[i];
            let lang = self.detect_lang(word);
            let offset = word_offsets[i];

            match lang {
                Lang::Punct => {
                    if let Some(&id) = self.token2id.get(word.as_str()) {
                        result.push(Token::new(id, word.clone(), offset));
                    } else {
                        // Never drop characters — use byte fallback
                        for (t, id) in self.byte_fallback(word) {
                            result.push(Token::new(id, t, offset));
                        }
                    }
                    // Punct doesn't change language state
                    i += 1;
                }
                Lang::Num => {
                    self.maybe_add_cs(&mut result, prev_lang, lang, offset);
                    if let Some(&id) = self.token2id.get(word.as_str()) {
                        result.push(Token::new(id, word.clone(), offset));
                    } else {
                        for (t, id) in self.byte_fallback(word) {
                            result.push(Token::new(id, t, offset));
                        }
                    }
                    prev_lang = Some(lang);
                    i += 1;
                }
                Lang::Code => {
                    self.maybe_add_cs(&mut result, prev_lang, lang, offset);
                    if let Some(&id) = self.token2id.get(word.as_str()) {
                        result.push(Token::new(id, word.clone(), offset));
                    } else {
                        let en_tokens = self.tokenize_en_word(word);
                        if en_tokens.iter().any(|(t, _)| t != &self.unk_token) {
                            for (t, id) in en_tokens {
                                result.push(Token::new(id, t, offset));
                            }
                        } else {
                            for (t, id) in self.byte_fallback(word) {
                                result.push(Token::new(id, t, offset));
                            }
                        }
                    }
                    prev_lang = Some(lang);
                    i += 1;
                }
                Lang::Vi => {
                    let mut run_end = i + 1;
                    while run_end < n && self.detect_lang(&words[run_end]) == Lang::Vi {
                        run_end += 1;
                    }
                    let run_raw: Vec<String> = words[i..run_end].to_vec();
                    let run_lower: Vec<String> = run_raw
                        .iter()
                        .map(|w| w.trim_matches(STRIP_CHARS).to_lowercase())
                        .collect();
                    let run_raw_refs: Vec<&str> = run_raw.iter().map(|s| s.as_str()).collect();
                    let run_offsets = &word_offsets[i..run_end];

                    self.maybe_add_cs(&mut result, prev_lang, Lang::Vi, offset);
                    self.tokenize_vi_run(
                        &run_lower,
                        &run_raw_refs,
                        run_offsets,
                        &mut result,
                    );
                    prev_lang = Some(Lang::Vi);
                    i = run_end;
                }
                Lang::En => {
                    self.maybe_add_cs(&mut result, prev_lang, Lang::En, offset);
                    let en_tokens = self.tokenize_en_word(word);
                    for (t, id) in &en_tokens {
                        result.push(Token::new(*id, t.clone(), offset));
                    }
                    prev_lang = Some(Lang::En);
                    i += 1;
                }
            }
        }

        Ok(result)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token2id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id2token.get(&id).cloned()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token2id.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.token2id.len()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let file_name = match name {
            Some(n) => format!("{n}-tokenizer.json"),
            None => "tokenizer.json".to_string(),
        };
        let path: PathBuf = [folder, Path::new(&file_name)].iter().collect();

        let sp_pieces: Vec<(String, f64)> = {
            let mut pairs: Vec<_> = self
                .sp_vocab
                .keys()
                .filter_map(|k| {
                    // Reconstruct from trie - simplified: just save what we have
                    Some((
                        k.clone(),
                        self.sp_trie
                            .children
                            .get(&k.chars().next().unwrap_or('\0'))
                            .and_then(|n| {
                                let mut node = n;
                                for c in k.chars().skip(1) {
                                    node = node.children.get(&c)?;
                                }
                                Some(node.score)
                            })
                            .unwrap_or(0.0),
                    ))
                })
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            pairs
        };

        let cfg = ViLLMConfig {
            type_: "ViLLM".to_string(),
            token2id: self.token2id.clone(),
            unk_token: self.unk_token.clone(),
            vi_syllables: self.vi_syllables.iter().cloned().collect(),
            base_forms: self.base_forms.iter().cloned().collect(),
            vi_compounds: self.vi_compounds.clone(),
            compound_unigram_score: self.compound_unigram_score,
            sp_pieces,
            cs_vi_en: self.cs_vi_en.clone(),
            cs_en_vi: self.cs_en_vi.clone(),
        };

        let json = serde_json::to_string_pretty(&cfg)?;
        std::fs::write(&path, json)?;
        Ok(vec![path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        crate::models::TrainerWrapper::BpeTrainer(crate::models::bpe::BpeTrainer::default())
    }
}

// ── Serialization ──────────────────────────────────────────────

impl Serialize for ViLLMModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let sp_pieces: Vec<(String, f64)> = self
            .sp_vocab
            .keys()
            .map(|k| {
                (
                    k.clone(),
                    self.sp_trie
                        .children
                        .get(&k.chars().next().unwrap_or('\0'))
                        .map(|n| {
                            let mut node = n;
                            for c in k.chars().skip(1) {
                                if let Some(child) = node.children.get(&c) {
                                    node = child;
                                } else {
                                    break;
                                }
                            }
                            node.score
                        })
                        .unwrap_or(0.0),
                )
            })
            .collect();

        let cfg = ViLLMConfig {
            type_: "ViLLM".to_string(),
            token2id: self.token2id.clone(),
            unk_token: self.unk_token.clone(),
            vi_syllables: self.vi_syllables.iter().cloned().collect(),
            base_forms: self.base_forms.iter().cloned().collect(),
            vi_compounds: self.vi_compounds.clone(),
            compound_unigram_score: self.compound_unigram_score,
            sp_pieces,
            cs_vi_en: self.cs_vi_en.clone(),
            cs_en_vi: self.cs_en_vi.clone(),
        };
        cfg.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ViLLMModel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let cfg = ViLLMConfig::deserialize(deserializer)?;
        Ok(Self::from_config(cfg))
    }
}

// ── Constructor helpers for Python bindings ────────────────────

impl ViLLMModel {
    pub fn new(
        token2id: HashMap<String, u32>,
        unk_token: String,
        vi_syllables: Vec<String>,
        base_forms: Vec<String>,
        vi_compounds: HashMap<String, f64>,
        compound_unigram_score: f64,
        sp_pieces: Vec<(String, f64)>,
        cs_vi_en: String,
        cs_en_vi: String,
    ) -> Self {
        let cfg = ViLLMConfig {
            type_: "ViLLM".to_string(),
            token2id,
            unk_token,
            vi_syllables,
            base_forms,
            vi_compounds,
            compound_unigram_score,
            sp_pieces,
            cs_vi_en,
            cs_en_vi,
        };
        Self::from_config(cfg)
    }
}
