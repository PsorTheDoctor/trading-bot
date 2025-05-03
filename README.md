# Prerequirements
1. (Note: It's highly recommended to create and use Python 3.12+ virtual environment beforehand: https://realpython.com/python-virtual-environments-a-primer/)
Install dependencies with command:
```
pip install -r requirements.txt
```
2. Download and install trader app:
    1. MetaTrader: https://www.metatrader5.com/en/download
    2. BossaFX: https://bossafx.pl/oferta/platforma/MT5#Desktop
3. Open trader application and create new demo account - at the end of `Open an Account` process, click on `Copy the registration information to clipboard`, paste and save somewhere the content of your clipboard.
4. Create a new configuration file in the same location as this `README.md` file, named:
    1. For MetaTrader5: `meta_trader_key.txt`
    2. For BossaFX: `bossa_trader_key.txt`
5. Fill properly content of configuration file based on data you stored in step 3 - assuming that your login is `XXX`, password is `YYY` and server is `ZZZ`, your file content should looks like this:

```txt
XXX
YYY
ZZZ
```

# Recommended IDE plugins:
1. SonarQube: https://www.sonarsource.com/products/sonarlint/ - it's using static code analysis to provide tips about best practises in codebase (like hints about too high function cognitive complexity or import only specific functions instead of whole file)
