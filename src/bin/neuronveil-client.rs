use log::debug;
use ndarray::array;
use neuronveil::message::Message;
use ring::rand::{SecureRandom, SystemRandom};
use s2n_quic::{client::Connect, Client};
use std::{error::Error, net::SocketAddr, path::Path, sync::Arc, time::Duration};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Start the logger
    flexi_logger::Logger::try_with_env()
        .unwrap()
        .start()
        .unwrap();

    debug!("Initialising the CSPRNG");
    let system_random = Arc::new(SystemRandom::new());
    let mut random_buffer = [0u8; 4];
    system_random.fill(&mut random_buffer).unwrap();

    let client = Client::builder()
        .with_tls(Path::new("cert.pem"))?
        .with_io("0.0.0.0:0")?
        .start()?;

    debug!("Attempting to connect...");
    let addr: SocketAddr = "127.0.0.1:1967".parse()?;
    let connect = Connect::new(addr).with_server_name("localhost");
    let connection = client.connect(connect).await?;

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

    let output = neuronveil::client::infer(
        (&outcoming_sender, &mut incoming_receiver),
        array![1f32, 1f32, -1f32, -1f32],
        system_random.as_ref(),
    )
    .await?;

    println!("Output: {:#}", output);

    // FIXME this shouldn't be needed
    tokio::time::sleep(Duration::from_millis(100)).await;

    // TODO remove this:
    // ensure the connection doesn't time out with inactivity
    // connection.keep_alive(true)?;

    // open a new stream and split the receiving and sending sides
    // let mut stream = connection.open_send_stream().await?;

    // // spawn a task that copies responses from the server to stdout
    // tokio::spawn(async move {
    //     let mut stdout = tokio::io::stdout();
    //     let _ = tokio::io::copy(&mut receive_stream, &mut stdout).await;
    // });

    // copy data from stdin and send it to the server
    // let mut stdin = tokio::io::stdin();
    // let mut reader: &[u8] = b"{}";
    // tokio::io::copy(&mut reader, &mut stream).await?;
    // stream.close().await?;

    Ok(())
}
