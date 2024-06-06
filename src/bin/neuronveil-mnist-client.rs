use anyhow::Context;
use clap::{command, Parser};
use image::{imageops::FilterType, io::Reader as ImageReader, GrayImage};
use log::debug;
use ndarray::{array, Array1};
use neuronveil::{message::Message, model::Model, Com};
use ring::rand::{SecureRandom, SystemRandom};
use s2n_quic::{client::Connect, Client};
use std::{
    error::Error, fs::File, io::BufReader, net::SocketAddr, path::Path, sync::Arc, time::Duration,
};
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input image of a hand-written digit to identify
    image: std::path::PathBuf,

    /// Infer a model locally, without connecting to any server
    model: Option<std::path::PathBuf>,

    /// Server IP address and port
    #[arg(long, default_value = "127.0.0.1:1967")]
    server: SocketAddr,

    /// Server name per the QUIC protocol
    #[arg(long, default_value = "localhost")]
    server_name: String,
}

fn load_image<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Array1<f32>> {
    // Read the image
    let raw_image = ImageReader::open(filepath)?.decode()?;

    // Resize the image to the model's input size and convert init to a greyscale buffer
    let image: GrayImage = raw_image
        .resize_exact(8, 8, FilterType::Nearest)
        .grayscale()
        .into();

    // Convert the image to an array with values in range [0, 16)
    Ok(Array1::from_iter(image.pixels().map(|v| v.0[0] as f32)) / 16.0)
}

async fn infer_online(
    input: Array1<f32>,
    server: SocketAddr,
    server_name: String,
) -> anyhow::Result<Array1<f32>> {
    // Connect to the server
    debug!("Attempting to connect to {}", server);
    let client = Client::builder()
        .with_tls(Path::new("cert.pem"))?
        .with_io("0.0.0.0:0")?
        .start()?;
    let connect = Connect::new(server).with_server_name(server_name);
    let connection = client.connect(connect).await?;

    debug!("Initialising the CSPRNG");
    let system_random = Arc::new(SystemRandom::new());
    let mut random_buffer = [0u8; 4];
    system_random.fill(&mut random_buffer).unwrap();

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
        input,
        system_random.as_ref(),
    )
    .await?;

    // FIXME this shouldn't be needed
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(output)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Start the logger
    flexi_logger::Logger::try_with_env()
        .unwrap()
        .start()
        .unwrap();

    let args = Args::parse();

    let input = load_image(args.image).context("Failed to load the input image")?;

    let output = if let Some(model) = args.model {
        // Convert the input from float to Com
        let input_com = input.mapv(Com::from_num);

        // Read the model file
        let model_file = File::open(model)?;
        let reader = BufReader::new(model_file);
        let model: Model = serde_json::from_reader(reader)?;

        // Infer locally
        let output_com = model.infer_locally(input_com);

        // Convert the output from Com to float
        output_com.mapv(Com::to_num::<f32>)
    } else {
        // Infer online, without knowing the model
        infer_online(input, args.server, args.server_name)
            .await
            .context("Online inference failed")?
    };

    println!(
        "Output: {:#}",
        neuronveil::utils::softmax(&output.view()).unwrap()
    );

    // // FIXME this shouldn't be needed
    // tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(())
}
