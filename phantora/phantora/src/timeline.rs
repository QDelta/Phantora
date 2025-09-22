use crate::event_queue::{Action, EventId};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::io::Write;
use std::path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Record {
    Action { id: EventId, action: Action },
    Span { id: EventId, start: i64, end: i64 },
}

impl Record {
    #[inline]
    pub fn new_span(id: EventId, start: i64, end: i64) -> Self {
        Record::Span { id, start, end }
    }

    #[inline]
    pub fn new_action(id: EventId, action: Action) -> Self {
        Record::Action { id, action }
    }
}

pub(crate) struct Timeline {
    writer: io::BufWriter<fs::File>,
}

impl Timeline {
    pub(crate) fn new<P: AsRef<path::Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::create(path)?;
        Ok(Self {
            writer: io::BufWriter::new(file),
        })
    }

    pub(crate) fn write(&mut self, rec: &Record) -> io::Result<()> {
        let len = bincode::serialized_size(rec).unwrap();
        let buf = bincode::serialize(rec).unwrap();
        self.writer.write(&len.to_be_bytes())?;
        self.writer.write(&buf)?;
        self.writer.flush()?;
        Ok(())
    }
}
