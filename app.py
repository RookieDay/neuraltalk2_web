import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import logging
import optparse
import tornado.wsgi
import tornado.httpserver
import sys
import time
import shutil
import json
import argparse
import base64
import urllib.request
# Initialize the Flask application
app = Flask(__name__)

base_dir = os.getcwd()
img_load = base_dir +'/images/'
app.config['UPLOAD_FOLDER'] = 'vis/imgs/'
app.config['JSON_PATH'] = 'vis/'
app.config['EVAL_PATH'] = ' eval.lua '
app.config['GPU_MODE'] = 'model_id1-501-1448236541.t7'
app.config['CPU_MODE'] = 'model_id1-501-1448236541.t7_cpu.t7'
app.config['IMG_PATH'] = 'images/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html',has_result=False)

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the name of the uploaded files
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        for file in uploaded_files:
            # Check if the file is one of the allowed types/extensions
            if file and allowed_file(file.filename):
                # Make the filename safe, remove unsupported chars
                filename = secure_filename(file.filename)
                # Move the file form the temporal folder to the upload
                # folder we setup
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Save the filename into a list, we'll use it later
                filenames.append(filename)
                # Redirect the user to the uploaded_file route, which
                # will basicaly show on the browser the uploaded file
        # Load an html page with a link to each uploaded file

    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        return render_template(
            'index.html',has_result=False
        ) 
    process_image, times_used  = app.clf.image_caption(filenames) 
    # sys.stdout.write(str(process_image)+'*****'+'\n')
    return render_template('upload.html', return_process=process_image, times = times_used)


@app.route('/vis/imgs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    try:
        imageurl = request.args.get('imageurl', '')
        # download
        user_Agent = 'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
        header = {'User-Agent':user_Agent}
        req = urllib.request.Request(imageurl,headers=header)
        raw_data = urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        logging.info('url image open error: %s', err)
        return render_template(
            'index.html',has_result=False
        ) 

    else:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.jpg')

        with open(filename,'wb') as f:
            f.write(raw_data)
        f_p , f_name = os.path.split(filename)
        process_image, times_used = app.clf.image_caption([f_name])
        return render_template('index.html', return_process=process_image, times = times_used, has_result=True)


class ImageCaption(object):
    # 预先加载参数
    def __init__(self, gpu_mode,model,img_path,num_images):
        logging.info('Loading net and associated files...')
        if gpu_mode:
           self.gpu_mode = True
           self.model = os.getcwd() + '/' + app.config['GPU_MODE'] 
        else:
            self.gpu_mode = False
            self.model = model
        self.img_path = img_path
        self.num_images = num_images


    def image_caption(self,filenames):
        process_image = []
        process_caption = []
        if os.path.exists(img_load):
            shutil.rmtree(img_load)
            os.makedirs(img_load)
        for file_ in filenames:
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], file_),os.path.join(app.config['IMG_PATH'], file_))
        starttime = time.time()
        # if app.clf.gpu_mode:
        #     os.system('th '+ app.config['EVAL_PATH'] + ' -model ' + app.clf.model  + ' -image_folder ' + app.clf.img_path + ' -num_images ' + app.clf.num_images)
        # else:
        #     os.system('th'+ app.config['EVAL_PATH'] + ' -model ' + app.clf.model + ' -image_folder ' + app.clf.img_path + ' -num_images ' + app.clf.num_images + ' -gpuid -1')
        endtime = time.time()

        with open(os.path.join(app.config['JSON_PATH'], 'vis.json'), 'r') as f:
            temp = json.loads(f.read())

        for img_msg in temp:
            img_embed = embed_image_html(os.getcwd() + '/' + app.config['UPLOAD_FOLDER'] + 'img'+ img_msg['image_id']+'.jpg')
            # process_image[img_embed] = img_msg['caption']
            process_image.append(img_embed)
            process_caption.append(img_msg['caption'])
    
        return (process_image,process_caption), '%.3f' % (endtime - starttime)

def embed_image_html(fileps):
    """Creates an image embedded in HTML base64 format."""
    with open(fileps, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return 'data:image/jpg;base64,' + str(encoded_string)[2:-1]


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
          "--debug",
          type="bool",
          default=False,
          help="Use debugger to track down bad values during training")
    parser.add_argument(
          "--port",
          type=int,
          default=5000,
          help="which port to serve content on")
    parser.add_argument(
          "--g",
          type="bool",
          default=False,
          help="Use GPU model during training")
    parser.add_argument(
          "--model",
          type=str,
          default= os.getcwd() + "/model_id1-501-1448236541.t7_cpu.t7",
          help="Directory for storing data")
    parser.add_argument(
          "--img_path",
          type=str,
          default= os.getcwd() + "/images",
          help="Directory for storing data")

    parser.add_argument(
          "--num_images",
          type=str,
          default= '10',
          help="Directory for storing data")

    FLAGS, unparsed = parser.parse_known_args()

    init_stateModel = {'gpu_mode':FLAGS.g,'model':FLAGS.model,'img_path':FLAGS.img_path,'num_images':FLAGS.num_images}
    app.clf = ImageCaption(**init_stateModel)

    if FLAGS.debug:
        app.run(debug=True, host='0.0.0.0', port=FLAGS.port)
    else:
        start_tornado(app, FLAGS.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)