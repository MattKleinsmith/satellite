# Clearing Jupyter cell labels

![docs/assets/clear_jupyter_labels.png](../docs/assets/clear_jupyter_labels.png)

Clearing Jupyter cell labels before commiting helps declutter commits.

To clear them from your .py file, add the following code to your `~/.bashrc` and type `ga your-file.py` in a terminal.

```bash
export satellite_repo="<the directory where you git cloned>"
export jupyter_clearer=$satellite_repo"dstl/meta/hooks/clear_jupyter_cell_labels.py"
function ga(){                                                                                       
    if [[ $1 == *.py ]]; then                                                                        
        python3 $jupyter_clearer `pwd`/$1                                                            
    fi                                                                                               
    git add $1                                                                                       
}  
```

"ga" stands for "git add". Feel free to remove the `git add` line if you just want to use the clearer.
