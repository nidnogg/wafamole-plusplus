ubuntu 16.04 wsl
python3.5 -m pip COMMAND to install anything on WSL

on debian vbox 
python3 setup.py build
python3 setup.py install
pip install -r requirements.txt
pip install scikit-learn==0.21.3

(!) new model
psudo pip install . && wafamole evade --model-type svc wafamole/models/custom/svc/svc_trained.dump  "admin' OR 1=1#"

psudo pip install . && wafamole evade --model-type waf-brain wafamole/models/custom/example_models/waf-brain.h5  "admin' OR 1=1#"

psudo pip install . && wafamole evade --model-type token wafamole/models/custom/example_models/naive_bayes_trained.dump  "admin' OR 1=1#"

(!) needs shut up module for deprecation warnings
psudo pip install . && wafamole evade --model-type token wafamole/models/custom/example_models/random_forest_trained.dump  "admin' OR 1=1#"

psudo pip install . && wafamole evade --model-type token wafamole/models/custom/example_models/lin_svm_trained.dump  "admin' OR 1=1#"

psudo pip install . && wafamole evade --model-type token wafamole/models/custom/example_models/gauss_svm_trained.dump  "admin' OR 1=1#"

SQLiGoT to be avoided - takes hours (?)
psudo pip install . && wafamole evade --model-type DP wafamole/models/custom/example_models/graph_directed_proportional_sqligot "admin' OR 1=1#"