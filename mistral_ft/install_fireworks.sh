apt-get install wget
wget -O firectl.gz https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz
gunzip firectl.gz
install -o root -g root -m 0755 firectl /usr/local/bin/firectl