# ИИ на страже популяции ненецких моржей
Реализация подготовлена командой RostovNats 

## Установка

### Через готовый образ Docker
Если не установлен Docker с поддержкой CUDA установите его с помощью следующей команды:
```shell
sh installation/install_cuda_docker.sh
```

### Вручную

Сборка Docker image
```shell
sudo docker build -t walruses_rostovnats_server .
```

Запуск Docker container
С подключением к терминалу контекнера:
```shell
sudo docker run -p 8001:8001 --name RostovNatsServer -it walruses_rostovnats_server
```

С запуском в фоне:
```shell
sudo docker run -p 8001:8001 --name RostovNatsServer -d walruses_rostovnats_server
```

Если нужно удалить контейнер:
```shell
sudo docker rm RostovNatsServer
```