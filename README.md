# ИИ на страже популяции ненецких моржей
Реализация подготовлена командой RostovNats 

## Системные требования
* GPU >= Nvidia GTX 1060 (requires 6 GB video memory)
* RAM >= 8 GB
* Disk space >= 25 GB
* OS >= Ububtu 20.04 with CUDA 13.1 (this implementation tested on ubuntu 24.04)

## Установка

### Через готовый образ Docker
Если не установлен Docker с поддержкой CUDA установите его с помощью следующей команды:
```shell
sh installation/install_cuda_docker.sh
```

Затем скачайте готовый образ с решением по следующей ссылке:
https://disk.yandex.ru/d/26E-hplXNTEU0w

А далее загрузите его в Docker:
```shell
sudo docker load --input walruses_rostovnats_server.tar
```

### Вручную

#### Через Docker
Сборка Docker image
```shell
sudo docker build -t walruses_rostovnats_server .
```

## Запуск
Запуск Docker container
С подключением к терминалу контекнера:
```shell
sudo docker run --runtime=nvidia -p 8501:8501 --name RostovNatsServer -it walruses_rostovnats_server
```

С запуском в фоне:
```shell
sudo docker run --runtime=nvidia -p 8501:8501 --name RostovNatsServer -d walruses_rostovnats_server
```

Если нужно удалить контейнер:
```shell
sudo docker rm RostovNatsServer
```

## Локальное использование
Для работы с приложением откройте у себя в браузере следующую страницу:
http://localhost:8501/