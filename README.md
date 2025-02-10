# Prerequirements
1. (Note: It's higly recommended to create and use Python virtual environment beforehand: https://realpython.com/python-virtual-environments-a-primer/)
Install dependencies with command:
```
pip install -r requirements.txt
```
2. Download and install MetaTrader: https://www.metatrader5.com/en/download
3. Open MetaTrader and create new demo account - at the end of `Open an Account` process, click on `Copy the registration information to clipboard`, paste and save somewhere the content of your clipboard.
4. Create a new file in the same location as this `README.md` file, named `meta_trader_key.txt`
5. Fill properly content of `meta_trader_key.txt` file based on data you stored in step 3 - assuming that your login is `XXX`, password is `YYY` and server is `ZZZ`, your file content should looks like this:

```txt filename="meta_trader_key.txt"
XXX
YYY
ZZZ
```

# Recommended IDE plugins:
1. SonarQube: https://www.sonarsource.com/products/sonarlint/ - it's using static code analysis to provide tips about best practises in codebase (like hints about too high function cognitive complexity or import only specific functions instead of whole file)
