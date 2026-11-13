//! The bounded admission queue between the HTTP front and the batcher (DESIGN.md §7). Admission never blocks:
//! a full queue is the server's backpressure signal (503).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Instant;

pub struct AdmissionQueue<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
    capacity: usize,
}

struct State<T> {
    items: VecDeque<T>,
    closed: bool,
}

/// The outcome of [`AdmissionQueue::pop`].
#[derive(Debug, PartialEq, Eq)]
pub enum Pop<T> {
    Item(T),
    /// The deadline passed with the queue empty.
    Timeout,
    /// The queue was closed and is empty.
    Closed,
}

impl<T> AdmissionQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "admission queue needs a capacity");
        AdmissionQueue { state: Mutex::new(State { items: VecDeque::new(), closed: false }), ready: Condvar::new(), capacity }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Admit every item or none: an `n`-request batch call is refused whole when fewer than `n` places are free
    /// (or the queue is closed), and the items come back.
    pub fn try_push_all(&self, items: Vec<T>) -> Result<(), Vec<T>> {
        let mut s = self.state.lock().unwrap();
        if s.closed || s.items.len() + items.len() > self.capacity {
            return Err(items);
        }
        s.items.extend(items);
        drop(s);
        self.ready.notify_one();
        Ok(())
    }

    /// The oldest item, waiting until `deadline` (forever when None) for one to arrive.
    pub fn pop(&self, deadline: Option<Instant>) -> Pop<T> {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(item) = s.items.pop_front() {
                return Pop::Item(item);
            }
            if s.closed {
                return Pop::Closed;
            }
            s = match deadline {
                None => self.ready.wait(s).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Pop::Timeout;
                    }
                    self.ready.wait_timeout(s, d - now).unwrap().0
                }
            };
        }
    }

    /// Refuse further admissions; [`pop`](Self::pop) drains what is queued, then reports `Closed`.
    pub fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn bounded_all_or_nothing_fifo() {
        let q = AdmissionQueue::new(3);
        q.try_push_all(vec![1, 2]).unwrap();
        assert_eq!(q.try_push_all(vec![3, 4]), Err(vec![3, 4]));
        q.try_push_all(vec![3]).unwrap();
        assert_eq!(q.try_push_all(vec![5]), Err(vec![5]));
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(None), Pop::Item(1));
        assert_eq!(q.pop(None), Pop::Item(2));
        assert_eq!(q.pop(None), Pop::Item(3));
        assert_eq!(q.pop(Some(Instant::now() + Duration::from_millis(5))), Pop::Timeout);
    }

    #[test]
    fn close_drains_then_reports_closed() {
        let q = AdmissionQueue::new(2);
        q.try_push_all(vec![7]).unwrap();
        q.close();
        assert_eq!(q.try_push_all(vec![8]), Err(vec![8]));
        assert_eq!(q.pop(None), Pop::Item(7));
        assert_eq!(q.pop(None), Pop::Closed);
    }

    #[test]
    fn pop_wakes_on_push() {
        let q = std::sync::Arc::new(AdmissionQueue::new(2));
        let q2 = q.clone();
        let t = std::thread::spawn(move || q2.pop(None));
        std::thread::sleep(Duration::from_millis(20));
        q.try_push_all(vec![42]).unwrap();
        assert_eq!(t.join().unwrap(), Pop::Item(42));
    }
}
