# server

This contains code to start the Grover predictions server.  First get checkpoints using `get_ckpt.sh`. Then you can do something like this:


``` 
export CUDA_VISIBLE_DEVICES=2 && source activate turingadvice && export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
nohup python run_server.py > run_server.txt &
```

You might have to run these commands if things don't work:
```
sudo iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 4998 -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 4998 -m conntrack --ctstate ESTABLISHED -j ACCEPT
```

# To test

curl -X POST -d '{"instances": [{"title": "I am trying to debug this code and its really hard.", "selftext": "test test", "subreddit": "Advice"},{"title": "I am trying to debug this code and its really hard.  airestn eairestn iarst iearnst ", "selftext": "test test", "subreddit": "Advice"}], "target": "advice"}' -H "Content-Type: application/json" localhost:5000/api/askbatch