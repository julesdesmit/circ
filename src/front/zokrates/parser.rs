//! Parsing and recursively loading ZoKrates.
//!
//! Based on the original ZoKrates parser, with extra machinery for recursive loading and locating
//! the standard library.

use zokrates_pest_ast as ast;

use log::debug;
use std::collections::HashMap;

use crate::circify::includer::Loader;
use rug::Integer;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use typed_arena::Arena;

/// Parse an inputs file where each line has format: `no-withespace integer`.
///
/// Permits blank lines and ignores non-separating whitespace.
///
/// ```ignore
/// x 5
/// x.y -7
/// ```
pub fn parse_inputs(p: PathBuf) -> HashMap<String, Integer> {
    let mut m = HashMap::default();
    for l in BufReader::new(File::open(p).unwrap()).lines() {
        let l = l.unwrap();
        let l = l.trim();
        if l.len() > 0 {
            let mut s = l.split_whitespace();
            let key = s.next().unwrap().to_owned();
            let value = Integer::from(Integer::parse_radix(&s.next().unwrap(), 10).unwrap());
            m.insert(key, value);
        }
    }
    m
}

/// A representation of the standard libary's location.
pub struct ZStdLib {
    path: PathBuf,
}

impl ZStdLib {
    /// Looks for a "ZoKrates/zokrates_stdlib/stdlib" path in some ancestor of the current
    /// directory.
    pub fn new() -> Self {
        let p = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert!(p.is_absolute());
        let stdlib_subdirs = vec![
            "ZoKrates/zokrates_stdlib/stdlib",
            "third_party/ZoKrates/zokrates_stdlib/stdlib",
        ];
        for a in p.ancestors() {
            for subdir in &stdlib_subdirs {
                let mut q = a.to_path_buf();
                q.push(subdir);
                if q.exists() {
                    return Self { path: q };
                }
            }
        }
        panic!("Could not find ZoKrates stdlib from {}", p.display())
    }
    /// Turn `child`, relative to `parent` (or to the standard libary!), into an absolute path.
    pub fn canonicalize(&self, parent: &Path, child: &str) -> PathBuf {
        debug!("Looking for {} from {}", child, parent.display());
        let paths = [parent.to_path_buf(), self.path.clone()];
        for mut p in paths {
            p.push(child);
            if p.extension().is_none() {
                p.set_extension("zok");
            }
            debug!("Checking {}", p.display());
            if p.exists() {
                return p;
            }
        }
        panic!("Could not find {} from {}", child, parent.display())
    }
    /// check if this path is the EMBED prototypes path
    pub fn is_embed<P: AsRef<Path>>(&self, p: P) -> bool {
        p.as_ref().starts_with(&self.path) && p.as_ref().file_stem().map(|s| s.to_str()).flatten() == Some("EMBED")
    }
}

/// A recrusive zokrates loader
pub struct ZLoad {
    sources: Arena<String>,
    stdlib: ZStdLib,
}

impl ZLoad {
    /// Make a new ZoKrates loader, looking for the standard library somewhere above the current
    /// dirdirectory. See [ZStdLib::new].
    pub fn new() -> Self {
        Self {
            sources: Arena::new(),
            stdlib: ZStdLib::new(),
        }
    }

    /// Recursively load a ZoKrates file.
    ///
    /// ## Returns
    ///
    /// Returns a map from file paths to parsed files.
    pub fn load<P: AsRef<Path>>(&self, p: &P) -> HashMap<PathBuf, ast::File> {
        self.recursive_load(p).unwrap()
    }
}

impl<'a> Loader for &'a ZLoad {
    type ParseError = ();
    type AST = zokrates_pest_ast::File<'a>;

    fn parse<P: AsRef<Path>>(&self, p: &P) -> Result<Self::AST, Self::ParseError> {
        let mut s = String::new();
        File::open(p).unwrap().read_to_string(&mut s).unwrap();
        debug!("Parsing: {}", p.as_ref().display());
        let s = self.sources.alloc(s);
        let ast = ast::generate_ast(s);
        if ast.is_err() {
            panic!("{}", ast.unwrap_err());
        }
        Ok(ast.unwrap())
    }
    fn includes<P: AsRef<Path>>(&self, ast: &Self::AST, p: &P) -> Vec<PathBuf> {
        let mut c = p.as_ref().to_path_buf();
        c.pop();
        ast.declarations
            .iter()
            .filter_map(|d| {
                if let ast::SymbolDeclaration::Import(i) = d {
                    let ext = match i {
                        ast::ImportDirective::Main(m) => &m.source.value,
                        ast::ImportDirective::From(m) => &m.source.value,
                    };
                    Some(self.stdlib.canonicalize(&c, ext))
                } else {
                    None
                }
            })
            .collect()
    }
}
