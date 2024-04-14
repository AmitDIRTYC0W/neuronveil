// src/bin/client.rs
use s2n_quic::{client::Connect, Client};
use std::{error::Error, path::Path, net::SocketAddr};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::builder()
        .with_tls(Path::new("cert.pem"))?
        .with_io("0.0.0.0:0")?
        .start()?;

    println!("Attempting to connect..."); // TODO debug!
    let addr: SocketAddr = "127.0.0.1:4433".parse()?;
    let connect = Connect::new(addr).with_server_name("localhost");
    let mut connection = client.connect(connect).await?;

    // ensure the connection doesn't time out with inactivity
    connection.keep_alive(true)?;

    // open a new stream and split the receiving and sending sides
    let mut stream = connection.open_send_stream().await?;

    // // spawn a task that copies responses from the server to stdout
    // tokio::spawn(async move {
    //     let mut stdout = tokio::io::stdout();
    //     let _ = tokio::io::copy(&mut receive_stream, &mut stdout).await;
    // });

    // copy data from stdin and send it to the server
    let mut stdin = tokio::io::stdin();
    tokio::io::copy(&mut stdin, &mut stream).await?;

    Ok(())
}
