//! `census::join` — the shared keyed-table layer of the MITM census
//! engines.
//!
//! Every MITM census enumerates one *side* of a configuration space,
//! buckets it by a *key*, and joins side entries against the
//! complementary key of a target. The shared table is a build-once
//! sorted multimap: one sort at build time, cache-predictable
//! binary-search probes at lookup, and free read-only sharing across
//! rayon workers — the reasons the join loops prefer it to hashing.

/// A build-once multimap: rows sorted by key, payload slices found by
/// binary search. Payload order within one key is unspecified.
pub struct SortedMultiMap<K, V> {
    keys: Vec<K>,
    vals: Vec<V>,
}

impl<K: Ord + Copy, V> SortedMultiMap<K, V> {
    /// Build from rows in any order.
    pub fn new(mut rows: Vec<(K, V)>) -> Self {
        rows.sort_unstable_by_key(|r| r.0);
        let mut keys = Vec::with_capacity(rows.len());
        let mut vals = Vec::with_capacity(rows.len());
        for (k, v) in rows {
            keys.push(k);
            vals.push(v);
        }
        SortedMultiMap { keys, vals }
    }

    /// The payloads filed under `key` (empty slice when absent).
    pub fn get(&self, key: &K) -> &[V] {
        let lo = self.keys.partition_point(|k| k < key);
        let hi = lo + self.keys[lo..].partition_point(|k| k <= key);
        &self.vals[lo..hi]
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_misses() {
        let m = SortedMultiMap::new(vec![(3u64, 'a'), (1, 'b'), (3, 'c'), (7, 'd')]);
        assert_eq!(m.len(), 4);
        assert_eq!(m.get(&1), &['b']);
        assert_eq!(m.get(&3).len(), 2);
        assert!(m.get(&0).is_empty() && m.get(&4).is_empty() && m.get(&9).is_empty());
    }
}
