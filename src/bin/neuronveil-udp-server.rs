use std::boxed::Box;
use std::fs::File;
use std::io;
use std::io::BufReader;

use log::{debug, info};
use ring::rand::{SecureRandom, SystemRandom};
// use rmp_serde as rmps;
use serde_json;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use neuronveil::message::Message;
use neuronveil::model::Model;
use neuronveil::split::Split;

#[tokio::main]
async fn main() -> io::Result<()> {
    info!("Initialising the CSPRNG");
    let system_random = SystemRandom::new();
    let mut random_buffer = [0u8; 4];
    system_random.fill(&mut random_buffer).unwrap();

    info!("Reading the model");
    let file = File::open("identity.json")?;
    let reader = BufReader::new(file);
    let model: Model = serde_json::from_reader(reader)?;

    info!("Splitting the model into shares");
    let model_shares = model.split(&system_random);

    let (outcoming_sender, mut outcoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number
    let (incoming_sender, mut incoming_receiver) = mpsc::channel(1024); // TODO 1024 is a magic number

    tokio::spawn(async move {
        loop {
            let message = outcoming_receiver.recv().await;

            // TODO use if let some...
            match message {
                Some(ref contents) => {
                    let text = serde_json::to_string(&contents).unwrap(); // TODO handle errors
                    println!("Sending: {}", text);
                }
                None => {}
            };
        }
    });

    tokio::spawn(async move {
        info!("Binding to UDP 0.0.0.0:1967");
        let socket = UdpSocket::bind("0.0.0.0:1967").await.unwrap(); // TODO handle error

        info!("Listening...");
        loop {
            let mut buf = Box::new([0; 1024]); // TODO use maybe uninit

            let (len, addr) = socket.recv_from(buf.as_mut_slice()).await.unwrap(); // TODO handle error
            println!("{:?} bytes received from {:?}", len, addr);

            let text = String::from_utf8(buf[0..len].to_vec()).unwrap(); // TODO handle errors
            let message: Message = serde_json::from_str(&text).unwrap();

            incoming_sender.send(message).await.unwrap();

            debug!("Sent!");

            // let len = socket.send_to(&buf[..len], addr).await?;
            // println!("{:?} bytes sent", len);
        }
    });

    info!("Starting inference");
    neuronveil::server::infer((&outcoming_sender, incoming_receiver), model_shares)
        .await
        .unwrap(); // TODO add ?

    Ok(())
}
