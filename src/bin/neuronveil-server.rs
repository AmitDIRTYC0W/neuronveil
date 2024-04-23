use std::sync::{Arc, Mutex};
use std::{error::Error, fs::File, io::BufReader, path::Path};

use flexi_logger;
use log::debug;
use ring::rand::SecureRandom;
use ring::rand::SystemRandom;
use s2n_quic::{connection::Connection, Server};
use tokio::sync::mpsc;

use neuronveil::message::Message;
use neuronveil::model::Model;
use neuronveil::split::Split;
use tokio::task;

// thread_local! {
//     static SYSTEM_RANDOM: RefCell<Option<SystemRandom>> = RefCell::new(None); // NOTE a Cell/RefCell might be needed
// }

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Start the logger
    flexi_logger::Logger::try_with_env()
        .unwrap()
        .start()
        .unwrap();

    debug!("Reading the model");
    let file = File::open("model.json")?;
    let reader = BufReader::new(file);
    let model: Model = serde_json::from_reader(reader)?;

    debug!("Starting the server");
    let mut server = Server::builder()
        .with_tls((Path::new("cert.pem"), Path::new("key.pem")))?
        .with_io("127.0.0.1:1967")?
        .start()?;

    let local = task::LocalSet::new();

    while let Some(connection) = server.accept().await {
        // FIXME this should use spawn, not spawn_local
        let local_model = model.clone();
        local
            .run_until(async move {
                tokio::task::spawn_local(handle_connection(connection, local_model)).await;
            })
            .await;
    }

    Ok(())
}

async fn handle_connection(connection: Connection, model: Model) {
    debug!("Initialising the task-local(!) CSPRNG");
    let system_random = SystemRandom::new();
    system_random.fill(&mut [0u8; 4]).unwrap();

    debug!("New connection from {}", connection.remote_addr().unwrap());

    let (mut connection_handle, mut stream_acceptor) = connection.split();

    // Prepare for listening
    let (incoming_sender, mut incoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number

    tokio::spawn(async move {
        while let Ok(Some(mut stream)) = stream_acceptor.accept_receive_stream().await {
            // Fully receive the message
            let mut buffer: Vec<u8> = vec![];
            tokio::io::copy(&mut stream, &mut buffer).await.unwrap();

            // Parse it
            let message: Message = serde_json::from_slice(&buffer).unwrap();
            debug!("Received a message: {:?}", message);

            // Process it
            incoming_sender.send(message).await.unwrap();
        }
    });

    // Prepare for sending
    let (outcoming_sender, mut outcoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number

    tokio::spawn(async move {
        while let Some(message) = outcoming_receiver.recv().await {
            let mut stream = connection_handle.open_send_stream().await.unwrap(); // TODO handle errors!

            let buffer = serde_json::to_vec(&message).unwrap().try_into().unwrap();
            debug!("Attempting to send a message!");

            stream.send(buffer).await.expect("stream should be open");
            stream.close().await.unwrap();
        }
    });

    // Split the model into shares
    // TODO This should be done in advance
    let model_shares = model.split(&system_random);

    // Start infering
    debug!("Starting the inference");
    neuronveil::server::infer(
        (&outcoming_sender, &mut incoming_receiver),
        model_shares,
        &system_random,
    )
    .await
    .unwrap(); // TODO add ?
}
