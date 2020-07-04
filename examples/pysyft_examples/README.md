### How to run this

From `./fedPlay` run the following command to start 3 clients.3
```bash
    python examples/pysyft_examples/create_and_run_pysyft_clients.py --num 3
```
It will start 3 clients `client0, client1, client2` :
```
    Namespace(cpu=0.2, mem=512, num=3, seed=42, start=5000)
    Starting container client0
    Starting container client1
    Starting container client2
    services:
      client0:
        container_name: client0
        cpus: '0.2'
        environment:
        - WORKER_ID=client0
        image: anirbandas/pysyft-worker
        mem_limit: 512m
        ports:
        - 5000:8777
      client1:
        container_name: client1
        cpus: '0.2'
        environment:
        - WORKER_ID=client1
        image: anirbandas/pysyft-worker
        mem_limit: 512m
        ports:
        - 5001:8777
      client2:
        container_name: client2
        cpus: '0.2'
        environment:
        - WORKER_ID=client2
        image: anirbandas/pysyft-worker
        mem_limit: 512m
        ports:
        - 5002:8777
    version: '2.2'

    Starting client1 ... done
    Starting client0 ... done
    Creating client2 ... done
    Attaching to client1, client0, client2
```