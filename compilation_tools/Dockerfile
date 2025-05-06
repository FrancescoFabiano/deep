FROM ubuntu:18.04
RUN apt-get update
RUN apt install -y python3.7
RUN apt-get update
RUN apt-get install -y build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev libboost-all-dev
RUN apt-get update
RUN apt-get install libboost-all-dev
RUN mkdir EFP
#aggiunto la cartella del progetto al container possibile effettuare anche un git pull ma è da installare git nel container
ADD EFP /EFP/.