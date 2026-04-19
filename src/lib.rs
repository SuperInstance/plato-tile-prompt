//! plato-tile-prompt — Tile Context → LLM Prompt Assembly
//!
//! Takes scored tiles and assembles them into a prompt for LLM inference.
//! Handles budget management, deadband injection, and format selection.
//!
//! ## Why
//! A model doesn't receive raw tiles. It receives a prompt assembled from
//! relevant tiles, formatted for its context window, with deadband warnings
//! injected when approaching negative space.
//!
//! ## API
//! ```ignore
//! let config = PromptConfig::default();
//! let (prompt, stats) = PromptAssembler::build(&scored_tiles, query, &config);
//! ```

/// A scored tile ready for prompt assembly.
#[derive(Debug, Clone)]
pub struct ScoredTile {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub score: f64,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Priority { P0, P1, P2 }
impl Default for Priority { fn default() -> Self { Priority::P2 } }

/// Prompt assembly configuration.
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// Maximum tokens for the assembled prompt (approximate, uses chars/4).
    pub max_tokens: usize,
    /// Include deadband warnings in the prompt.
    pub inject_deadband: bool,
    /// Format style for tiles.
    pub format: TileFormat,
    /// System prompt prefix.
    pub system_prefix: String,
    /// Whether to include domain tags.
    pub include_domain: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        PromptConfig {
            max_tokens: 4096,
            inject_deadband: true,
            format: TileFormat::Structured,
            system_prefix: String::new(),
            include_domain: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TileFormat {
    /// Q: ... A: ... format
    Structured,
    /// Markdown with headers
    Markdown,
    /// JSON array
    Json,
    /// Compact: id: score | question → answer
    Compact,
}

/// The prompt assembler.
pub struct PromptAssembler;

impl PromptAssembler {
    /// Build a prompt from scored tiles.
    /// Returns (prompt, stats) where stats tracks what was included/excluded.
    pub fn build(tiles: &[ScoredTile], query: &str, config: &PromptConfig) -> (String, BuildStats) {
        let mut stats = BuildStats::default();
        let mut parts = Vec::new();

        // System prefix
        if !config.system_prefix.is_empty() {
            parts.push(config.system_prefix.clone());
            stats.system_tokens = estimate_tokens(&config.system_prefix);
        }

        // Sort by priority (P0 first) then score descending
        let mut sorted: Vec<&ScoredTile> = tiles.iter().collect();
        sorted.sort_by(|a, b| {
            let pa = match a.priority { Priority::P0 => 0, Priority::P1 => 1, Priority::P2 => 2 };
            let pb = match b.priority { Priority::P0 => 0, Priority::P1 => 1, Priority::P2 => 2 };
            pa.cmp(&pb).then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
        });

        let budget = config.max_tokens.saturating_sub(stats.system_tokens);
        let mut used = 0usize;

        for tile in &sorted {
            let formatted = Self::format_tile(tile, config);
            let tile_tokens = estimate_tokens(&formatted);
            if used + tile_tokens > budget {
                stats.excluded += 1;
                continue;
            }
            parts.push(formatted);
            used += tile_tokens;
            stats.tiles_included += 1;
            stats.tile_tokens += tile_tokens;
            match tile.priority {
                Priority::P0 => stats.p0_count += 1,
                Priority::P1 => stats.p1_count += 1,
                Priority::P2 => stats.p2_count += 1,
            }
        }

        // Deadband injection
        if config.inject_deadband {
            let deadband = Self::deadband_section(tiles, &stats);
            if !deadband.is_empty() {
                let db_tokens = estimate_tokens(&deadband);
                parts.push(deadband);
                stats.deadband_tokens = db_tokens;
            }
        }

        // Query suffix
        let query_line = format!("\n\nQuery: {}", query);
        stats.query_tokens = estimate_tokens(&query_line);
        parts.push(query_line);

        stats.total_tokens = stats.system_tokens + stats.tile_tokens + stats.deadband_tokens + stats.query_tokens;

        (parts.join("\n\n"), stats)
    }

    fn format_tile(tile: &ScoredTile, config: &PromptConfig) -> String {
        match config.format {
            TileFormat::Structured => {
                let domain_tag = if config.include_domain { format!("[{}]", tile.domain) } else { String::new() };
                format!("{}Q: {}\nA: {}", domain_tag, tile.question, tile.answer)
            }
            TileFormat::Markdown => {
                let domain_tag = if config.include_domain { format!(" ({})", tile.domain) } else { String::new() };
                format!("## {}{}\n\n{}", tile.question, domain_tag, tile.answer)
            }
            TileFormat::Json => {
                format!(r#"{{"id":"{}","q":"{}","a":"{}","score":{:.3}}}"#, tile.id, tile.question, tile.answer, tile.score)
            }
            TileFormat::Compact => {
                format!("{}: {:.2} | {} → {}", tile.id, tile.score, tile.question, tile.answer)
            }
        }
    }

    /// Generate deadband warnings for domains NOT covered by included tiles.
    fn deadband_section(tiles: &[ScoredTile], stats: &BuildStats) -> String {
        if stats.tiles_included == 0 { return String::new(); }
        let covered_domains: std::collections::HashSet<&str> =
            tiles.iter().take(stats.tiles_included).map(|t| t.domain.as_str()).collect();
        if covered_domains.is_empty() { return String::new(); }

        // Count P0 negatives (tiles that SHOULD be here but aren't)
        let p0_negatives: Vec<&str> = tiles.iter()
            .filter(|t| t.priority == Priority::P0 && !covered_domains.contains(t.domain.as_str()))
            .map(|t| t.domain.as_str())
            .collect();

        if p0_negatives.is_empty() { return String::new(); }

        let unique: std::collections::HashSet<&str> = p0_negatives.into_iter().collect();
        let warnings: Vec<String> = unique.iter().map(|d| format!("- ⚠️ P0 gap: no coverage for [{}]", d)).collect();
        format!("## Deadband Warnings\n\n{}", warnings.join("\n"))
    }
}

/// Build statistics.
#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub tiles_included: usize,
    pub excluded: usize,
    pub p0_count: usize,
    pub p1_count: usize,
    pub p2_count: usize,
    pub system_tokens: usize,
    pub tile_tokens: usize,
    pub deadband_tokens: usize,
    pub query_tokens: usize,
    pub total_tokens: usize,
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 // rough: 1 token ≈ 4 chars
}

fn make_tile(id: &str, q: &str, a: &str, domain: &str, score: f64, priority: Priority) -> ScoredTile {
    ScoredTile { id: id.into(), question: q.into(), answer: a.into(), domain: domain.into(), score, priority }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_assembly() {
        let tiles = vec![
            make_tile("t1", "What is PLATO?", "Training pipeline.", "plato", 0.9, Priority::P2),
        ];
        let config = PromptConfig::default();
        let (prompt, stats) = PromptAssembler::build(&tiles, "tell me about PLATO", &config);
        assert!(prompt.contains("What is PLATO?"));
        assert!(prompt.contains("Training pipeline."));
        assert!(prompt.contains("tell me about PLATO"));
        assert_eq!(stats.tiles_included, 1);
        assert_eq!(stats.excluded, 0);
    }

    #[test]
    fn test_priority_sorting() {
        let tiles = vec![
            make_tile("p2", "Low priority", "Answer P2", "misc", 0.5, Priority::P2),
            make_tile("p1", "Medium priority", "Answer P1", "safety", 0.7, Priority::P1),
            make_tile("p0", "High priority", "Answer P0", "critical", 0.3, Priority::P0),
        ];
        let config = PromptConfig::default();
        let (prompt, stats) = PromptAssembler::build(&tiles, "test", &config);
        // P0 should appear first despite lowest score
        let p0_pos = prompt.find("High priority").unwrap();
        let p1_pos = prompt.find("Medium priority").unwrap();
        let p2_pos = prompt.find("Low priority").unwrap();
        assert!(p0_pos < p1_pos);
        assert!(p1_pos < p2_pos);
        assert_eq!(stats.p0_count, 1);
        assert_eq!(stats.p1_count, 1);
        assert_eq!(stats.p2_count, 1);
    }

    #[test]
    fn test_budget_exclusion() {
        let tiles = vec![
            make_tile("big", &"A".repeat(20000), &"B".repeat(20000), "x", 0.9, Priority::P2),
            make_tile("small", "Q?", "A.", "y", 0.5, Priority::P2),
        ];
        let config = PromptConfig { max_tokens: 100, ..Default::default() };
        let (_, stats) = PromptAssembler::build(&tiles, "test", &config);
        assert!(stats.excluded >= 1);
        assert!(stats.tiles_included >= 1);
    }

    #[test]
    fn test_deadband_injection() {
        let tiles = vec![
            make_tile("safe", "Safe topic", "Safe answer", "safe_domain", 0.9, Priority::P2),
        ];
        let config = PromptConfig { inject_deadband: true, ..Default::default() };
        let (prompt, stats) = PromptAssembler::build(&tiles, "test", &config);
        // No P0 gaps → no deadband warnings
        assert_eq!(stats.deadband_tokens, 0);
    }

    #[test]
    fn test_deadband_p0_gap() {
        // Small budget excludes the P0 tile, creating a gap
        let tiles = vec![
            make_tile("safe", "Safe topic question here", "Safe answer here", "safe_domain", 0.9, Priority::P2),
            make_tile("gap", "Critical safety question that is very long and will not fit in the tiny budget window provided here for testing purposes", "Critical answer also very long", "unsafe_domain", 0.1, Priority::P0),
        ];
        let config = PromptConfig { max_tokens: 50, inject_deadband: true, ..Default::default() };
        let (prompt, stats) = PromptAssembler::build(&tiles, "test", &config);
        // P0 tile excluded by budget → deadband warning should fire
        assert!(stats.excluded >= 1);
        assert!(stats.deadband_tokens > 0);
        assert!(prompt.contains("Deadband Warnings"));
    }

    #[test]
    fn test_no_deadband_when_disabled() {
        let tiles = vec![
            make_tile("gap", "Missing", "None", "gap_domain", 0.1, Priority::P0),
        ];
        let config = PromptConfig { inject_deadband: false, ..Default::default() };
        let (prompt, stats) = PromptAssembler::build(&tiles, "test", &config);
        assert_eq!(stats.deadband_tokens, 0);
        assert!(!prompt.contains("Deadband"));
    }

    #[test]
    fn test_structured_format() {
        let tiles = vec![make_tile("t1", "Q?", "A.", "dom", 0.9, Priority::P2)];
        let config = PromptConfig { format: TileFormat::Structured, include_domain: true, ..Default::default() };
        let (prompt, _) = PromptAssembler::build(&tiles, "test", &config);
        assert!(prompt.contains("[dom]"));
        assert!(prompt.contains("Q: Q?"));
        assert!(prompt.contains("A: A."));
    }

    #[test]
    fn test_markdown_format() {
        let tiles = vec![make_tile("t1", "What is flux?", "Bytecode runtime.", "flux", 0.9, Priority::P2)];
        let config = PromptConfig { format: TileFormat::Markdown, include_domain: true, ..Default::default() };
        let (prompt, _) = PromptAssembler::build(&tiles, "test", &config);
        assert!(prompt.contains("## What is flux?"));
        assert!(prompt.contains("(flux)"));
    }

    #[test]
    fn test_compact_format() {
        let tiles = vec![make_tile("t1", "Q?", "A.", "d", 0.85, Priority::P2)];
        let config = PromptConfig { format: TileFormat::Compact, ..Default::default() };
        let (prompt, _) = PromptAssembler::build(&tiles, "test", &config);
        assert!(prompt.contains("t1: 0.85"));
        assert!(prompt.contains("Q? → A."));
    }

    #[test]
    fn test_json_format() {
        let tiles = vec![make_tile("t1", "Q?", "A.", "d", 0.9, Priority::P2)];
        let config = PromptConfig { format: TileFormat::Json, ..Default::default() };
        let (prompt, _) = PromptAssembler::build(&tiles, "test", &config);
        assert!(prompt.contains(r#""id":"t1""#));
        assert!(prompt.contains(r#""score":0.900"#));
    }

    #[test]
    fn test_system_prefix() {
        let tiles = vec![make_tile("t1", "Q?", "A.", "d", 0.9, Priority::P2)];
        let config = PromptConfig { system_prefix: "You are a helpful PLATO assistant.".into(), ..Default::default() };
        let (prompt, stats) = PromptAssembler::build(&tiles, "test", &config);
        assert!(prompt.starts_with("You are a helpful PLATO assistant."));
        assert!(stats.system_tokens > 0);
    }

    #[test]
    fn test_empty_tiles() {
        let config = PromptConfig::default();
        let (prompt, stats) = PromptAssembler::build(&[], "test", &config);
        assert!(prompt.contains("Query: test"));
        assert_eq!(stats.tiles_included, 0);
    }

    #[test]
    fn test_token_accounting() {
        let tiles = vec![
            make_tile("t1", "Short Q", "Short A", "d", 0.9, Priority::P2),
        ];
        let config = PromptConfig::default();
        let (_, stats) = PromptAssembler::build(&tiles, "test query", &config);
        assert!(stats.total_tokens > 0);
        assert_eq!(stats.total_tokens, stats.system_tokens + stats.tile_tokens + stats.deadband_tokens + stats.query_tokens);
    }
}
