# SmartServe.Server

## Usage

Start a server:
- `--model-folder`(optional): The folder of model workspace. When receving a request specifying the model, it search firstly in this directory. For example, when I specify `--model-folder model`, it will search models in the directoy **./model** if it exists. When I specify the model folder with such a request: 
    ```json
    {
        "model": "llama3.1-8b-instruct",
        "prompt": "Hello"
    }
    ```
    it will search the directory **./model/llama3.1-8b-instruct** at first and then **llama3.1-8b-instruct** if the former one does not exist.
- `--host`(optional): The IP address the server listen
- `--port`(optional): The IP port the server listen
```shell
./build/bin/server --model-folder model --host 127.0.0.1 --port 8080
```

Test the server simply:
- Completion
    ```shell
    curl --request POST \                                              
        --url http://localhost:8080/completion \          
        --header "Content-Type: application/json" \            
        --data '{"prompt": "Once upon a time", "max_tokens": 128, "model": "model"}'
    ```
- Chat
    ```shell
    curl --request POST \
        --url http://localhost:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{"messages": [{"role":"user", "content":"Once upon a time"}], "model": "llama3.1-8b-q8"}'
    ```
- Streamly chat
    ```shell
    curl --request POST \
        --url http://localhost:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{"stream": true, "messages": [{"role":"user", "content":"Once upon a time"}], "model": "llama3.1-8b-q8"}'
    ```

## OpenAI API

SmartServe.server support part of OpenAI API:
- `/v1/completions`
- `/v1/chat/completions`
- `/v1/models`

As for the detail please refer to the [API documentation](https://platform.openai.com/docs/api-reference)
