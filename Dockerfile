FROM anirbandas/fedplay-base-python3:0.1
USER app
WORKDIR /home/app
COPY requirements.txt .
RUN pip install --user  --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","-u","examples/mlclient.py"]
