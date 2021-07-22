# Emotional Support

## <mark>Do not use Python 3.8 or later.</mark>

## Clone this repo & initialize

```sh
git clone https://gitlab.com/stvhuang/emo-sup.git
cd emo-sup
git submodule update --init --recursive
```

## Create Python Virtual Environment

See [link](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) for more information.

```sh
python3 -m venv .env
source .env/bin/activate
```

## Install required Python packages

```sh
pip install -r requirements.txt
```

## Install pyltp (https://github.com/HIT-SCIR/pyltp)

Apply this [PR](https://github.com/HIT-SCIR/pyltp/pull/193/files), and install from source.

```sh
cd pyltp
# apply the above PR
python setup.py install
cd ..
```

## Install bayonet (https://github.com/mpatacchiola/bayonet)

Install from source.

```sh
cd bayonet
make compile
ln -s ./libbayonet.so.1.0 bin/lib/libbayonet.1
cd ..
sourde ld_lib.sh
```

## Run!

```
python InteractionServer.py (2 for grpc version, 3 for web.py version)
```
## Demo

https://hackmd.io/E236ZMXbTiCzKc5ErZ9VjA?view

----
