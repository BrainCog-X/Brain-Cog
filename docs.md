# Sphinx 文档教程 

## 安装  

131 braincog 环境已经装好了
```shell
    pip install sphinx sphinx-rtd-theme recommonmark
``` 

## 配置 

已经配置好了, 直接用就行了
```shell
    sphinx-quickstart
``` 

## 编译

### braincog 之中的, 编译在Brain Docs之中

1. 重新从 repo 中抓取 ```rst``` 文本
```shell 
    cd braincog/docs 
    rm -rf ./source/braincog*rst
    sphinx-apidoc -o ./source/ ../braincog -f 
```

2. 编译 html 

```shell

    make clean
    make html 
```

### Examples 的编译 

1. 在 ```braincog/docs/source/index.rst``` 中, ```img_cls/Tutorial``` 后面一行添加 ``xxx/Tutorial``.
2. 然后在 ``Brain/docs/source`` 下面添加 ``xxx.md`` 文件, 要和上面的 ``xxx`` 同名.
3. 用 [Markdown](https://markdown.com.cn/basic-syntax/) 语法, 编写教程, 怎么用, 效果是啥. 
4. 编译html 

```shell
    make clean 
    make html
```

## 查看 

编译好的文件可以在 ```braincog/docs/build/html``` 中查看. 

## 上传

在130服务器上面:

```shell 
    sudo cp braincog/docs/build/html/* /var/www/html
```
就可以更新文档了, 并在 [172.18.116.130](http://172.18.116.130/index.html) 中看到.
