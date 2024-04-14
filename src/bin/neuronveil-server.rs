use std::{error::Error, path::Path, io::BufReader, fs::File, sync::Arc};

use flexi_logger;
use s2n_quic::{Server, connection::Connection};
use log::{debug, info, warn};
use ring::rand::{SecureRandom, SystemRandom};
use tokio::sync::mpsc;

use neuronveil::model::Model;
use neuronveil::split::Split;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Start the logger
    flexi_logger::Logger::try_with_env().unwrap().start().unwrap();

    debug!("Initialising the CSPRNG");
    let system_random = Arc::new(SystemRandom::new());
    let mut random_buffer = [0u8; 4];
    system_random.fill(&mut random_buffer).unwrap();

    debug!("Reading the model");
    let file = File::open("identity.json")?;
    let reader = BufReader::new(file);
    let model: Model = serde_json::from_reader(reader)?;

    debug!("Starting the server");
    let mut server = Server::builder()
        .with_tls((Path::new("cert.pem"), Path::new("key.pem")))?
        .with_io("127.0.0.1:4433")?
        .start()?;

    while let Some(mut connection) = server.accept().await {
        tokio::spawn(handle_connection(connection, model.clone(), Arc::clone(&system_random)));
    }

    Ok(())
}

async fn handle_connection(mut connection: Connection, model: Model, system_random: Arc<SystemRandom>) {
    // run in parallel:
    // connection.accept_receive_stream
    // sending loop

    debug!("New connection from {}", connection.remote_addr().unwrap());

    let (outcoming_sender, mut outcoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number
    let (incoming_sender, mut incoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number

    // Prepare for listening
    tokio::spawn(async move {
        while let Ok(Some(mut stream)) = connection.accept_receive_stream().await {
            debug!("Accepted a new stream");
            while let Ok(Some(data)) = stream.receive().await {
                debug!("Received some data");
            }
        }
    });

    // Prepare for sending
    // todo!();
    // TODO

    // Split the model into shares
    // TODO This should be done in advance
    let model_shares = model.split(system_random.as_ref());

    // Start infering
    debug!("Starting the inference");
    neuronveil::server::infer((&outcoming_sender, incoming_receiver), model_shares).await.unwrap(); // TODO add ?

    // while let Ok(Some(mut stream)) = connection.accept_bidirectional_stream().await {
    //     // spawn a new task for the stream
    //     debug!("Accepting a new stream");
    //     tokio::spawn(async move {
    //         // echo any data back to the stream
    //         while let Ok(Some(data)) = stream.receive().await {
    //             stream.send(data).await.expect("stream should be open");
    //         }
    //     });
    // }
}

