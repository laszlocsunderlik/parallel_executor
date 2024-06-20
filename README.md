**Build your docker image**
```
docker build -t your_image_name .
```
**Run the Docker container to mount local folders**
```
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output your_image_name
```