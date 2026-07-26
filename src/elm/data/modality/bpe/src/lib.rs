use fxhash::FxHashMap;
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::{prelude::*, ThreadPoolBuilder};

const PARALLEL_THRESHOLD: usize = 16_384;

type Pair = (u32, u32);
type MergeRule = (u32, u32, u32);
type PairCounts = FxHashMap<Pair, usize>;

#[inline]
fn merge(ids: &mut Vec<u32>, pair: Pair, new_id: u32) {
    let mut read = 0;
    let mut write = 0;

    while read < ids.len() {
        if read + 1 < ids.len() && (ids[read], ids[read + 1]) == pair {
            ids[write] = new_id;
            read += 2;
        } else {
            ids[write] = ids[read];
            read += 1;
        }

        write += 1;
    }

    ids.truncate(write);
}

fn get_stats(ids: &[u32]) -> PairCounts {
    if ids.len() < 2 {
        return PairCounts::default();
    }

    if ids.len() < PARALLEL_THRESHOLD {
        let mut counts = PairCounts::default();

        for window in ids.windows(2) {
            *counts.entry((window[0], window[1])).or_default() += 1;
        }

        return counts;
    }

    ids.par_windows(2)
        .fold(PairCounts::default, |mut counts, window| {
            *counts.entry((window[0], window[1])).or_default() += 1;
            counts
        })
        .reduce(PairCounts::default, |mut left, mut right| {
            if left.len() < right.len() {
                std::mem::swap(&mut left, &mut right);
            }

            for (pair, count) in right {
                *left.entry(pair).or_default() += count;
            }

            left
        })
}

fn best_pair(counts: &PairCounts) -> Option<Pair> {
    let mut best = None;

    for (&pair, &count) in counts {
        if best.map_or(true, |(best_pair, best_count)| {
            count > best_count || (count == best_count && pair < best_pair)
        }) {
            best = Some((pair, count));
        }
    }

    best.map(|(pair, _)| pair)
}

#[pyfunction]
fn byte_pair_encoding(
    text: &str,
    num_merges: usize,
    num_threads: usize,
) -> PyResult<(Vec<u32>, Vec<Vec<u8>>, Vec<MergeRule>)> {
    if num_threads == 0 {
        return Err(PyValueError::new_err(
            "num_threads must be greater than zero",
        ));
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();
    let mut vocab: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
    let mut merges = Vec::new();

    for _ in 0..num_merges {
        let counts = pool.install(|| get_stats(&ids));

        let Some((left, right)) = best_pair(&counts) else {
            break;
        };

        let new_id = u32::try_from(vocab.len())
            .map_err(|_| PyValueError::new_err("vocabulary exceeds u32 token IDs"))?;

        merge(&mut ids, (left, right), new_id);

        let mut token = vocab[left as usize].clone();
        token.extend_from_slice(&vocab[right as usize]);

        vocab.push(token);
        merges.push((left, right, new_id));
    }

    Ok((ids, vocab, merges))
}

#[pyfunction]
fn encode_symbol(text: &str, merges: Vec<MergeRule>) -> Vec<u32> {
    let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();

    let ranks: FxHashMap<Pair, (usize, u32)> = merges
        .into_iter()
        .enumerate()
        .map(|(rank, (left, right, new_id))| ((left, right), (rank, new_id)))
        .collect();

    loop {
        let mut best = None;

        for window in ids.windows(2) {
            let pair = (window[0], window[1]);

            if let Some(&(rank, new_id)) = ranks.get(&pair) {
                if best.map_or(true, |(best_rank, _, _)| rank < best_rank) {
                    best = Some((rank, pair, new_id));
                }
            }
        }

        let Some((_, pair, new_id)) = best else {
            break;
        };

        merge(&mut ids, pair, new_id);
    }

    ids
}

#[pyfunction]
fn decode_symbol(ids: Vec<u32>, vocab: Vec<Vec<u8>>) -> PyResult<String> {
    let mut bytes = Vec::new();

    for id in ids {
        let token = vocab
            .get(id as usize)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown token ID: {id}")))?;

        bytes.extend_from_slice(token);
    }

    String::from_utf8(bytes).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn bpe(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(byte_pair_encoding, module)?)?;
    module.add_function(wrap_pyfunction!(encode_symbol, module)?)?;
    module.add_function(wrap_pyfunction!(decode_symbol, module)?)?;
    Ok(())
}