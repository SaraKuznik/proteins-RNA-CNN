FROM tensorflow/tensorflow:1.10.1-gpu-py3

RUN pip3 --no-cache-dir install scikit-learn scikit-fusion matplotlib nimfa

# TensorBoard & Jupyter Notebook
EXPOSE 6006 8888

WORKDIR "/mag/"

# Jupyter has issues with being run directly: 
#   https://github.com/ipython/ipython/issues/7062 
# We just add a little wrapper script. 
COPY run_jupyter.sh /
RUN chmod +x /run_jupyter.sh

CMD ["/run_jupyter.sh", "--allow-root"]
