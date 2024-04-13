use std::boxed::Box;
use std::fs::File;
use std::io;
use std::io::BufReader;

use ndarray::{array, Array1};
use log::{debug, info};
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::json;
use tokio::net::UdpSocket;

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

    let socket = UdpSocket::bind("0.0.0.0:1967").await?;

    loop {
        let mut buf = Box::new([0; 1024]); // TODO use maybe uninit

        let (len, addr) = socket.recv_from(buf.as_mut_slice()).await?;
        println!("{:?} bytes received from {:?}", len, addr);

        // let len = socket.send_to(&buf[..len], addr).await?;
        // println!("{:?} bytes sent", len);
    }
}
