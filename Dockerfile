FROM anirbandas/fedplay-base-python3
WORKDIR /home/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY node .
CMD ["python","-u","node/server.py"]