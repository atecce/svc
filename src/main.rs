use book::Book;
use source::Source;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use gliner::model::pipeline::token::TokenMode;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

fn main() -> Result<()> {
    let word = bible::io::read_all();

    println!("initiating model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        "/Users/atec/etc/tokenizer.json",
        "/Users/atec/etc/model.onnx",
    )?;

    let mut index = HashMap::<String, Vec<Source>>::new();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("setting ctrlc handler");

    for (book, chapter_and_verse) in &word {
        for (i, chapter) in chapter_and_verse.iter().enumerate() {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            println!("initiating input for {} {}...", book, i + 1);
            let input = TextInput::new(chapter.to_vec(), vec![String::from("person")])?;

            println!("inferring...");
            let output = model.inference(input)?;
            for spans in output.spans {
                for span in spans {
                    println!(
                        "{:3} | {:16} | {:10} | {:.1}%",
                        span.sequence() + 1,
                        span.text(),
                        span.class(),
                        span.probability() * 100.0,
                    );

                    let src = Source {
                        book: Book { name: *book },
                        chapter: (i + 1).try_into().unwrap(),
                        verses: [(span.sequence() + 1).try_into().unwrap(), (span.sequence() + 1).try_into().unwrap()],
                    };

                    if let Some(srcs) = index.get_mut(span.text()) {
                        srcs.push(src);
                    } else {
                        index.insert(span.text().to_string(), vec![src]);
                    }
                }
            }
        }
    }

    let f = File::create("/Users/atec/index.json")?;
    let w = BufWriter::new(f);
    serde_json::to_writer(w, &index)?;

    Ok(())
}
