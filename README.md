# asp2sat
treewidth-aware reduction from asp to sat

## Requirements
We include a setup bash script `setup.sh` that should automatically perform all steps required to run our code. (Except for providing the c2d binary)

### Python
* Python >= 3.6

All required modules are listed in `requirements.txt` and can be obtained by running
```
pip install -r requirements.txt
```

### Treedecompositions via htd
We use [htd](https://github.com/TU-Wien-DBAI/htd) to obtain treedecompositions that are needed for our treedecomposition guided clark completion and for obtaining treewidth upperbounds on the programs.

It is included as a git submodule, together with [dpdb](https://github.com/hmarkus/dp_on_dbs) and [htd_validate](https://github.com/raki123/htd_validate). They are needed to parse the treedecompositions produced by htd.

The submodules can be obtained by running
```
git submodule update --init
```

htd further needs to be compiled. Detailed instructions can be found [here](https://github.com/mabseher/htd/blob/master/INSTALL.md) but in all likelihood it is enough to run
```
cd lib/htd/
cmake .
make -j8
```

## Usage

The basic usage is

```
python bin/main.py [OPTIONS] [<INPUT-FILES>]
```

The following options are accepted:
* `-global_mappings`: use global level mappings instead of local level mappings; can provide a performance boost on tree decompositions for large instances


## Credits

Acknowledgements go to Rafael Kiesel for implementing parts of the translator for a different context.
