import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import logging
import optparse
import tornado.wsgi
import tornado.httpserver
import sys
import time
import json
import shutil      
# Initialize the Flask application
app = Flask(__name__)

base_dir = os.getcwd()
img_load = base_dir + '/' + 'images/'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['JSON_PATH'] = 'vis/'
app.config['EVAL_PATH'] = ' eval.lua '
app.config['GPU_MODE'] = 'model_id1-501-1448236541.t7'
app.config['CPU_MODE'] = 'model_id1-501-1448236541.t7_cpu.t7'
app.config['IMG_PATH'] = 'images/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def image_caption(filenames):
    if os.path.exists(img_load):
        shutil.rmtree(img_load)
        os.makedirs(img_load)
    for file_ in filenames:
        shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], file_),os.path.join(app.config['IMG_PATH'], file_))
    # starttime = time.time()
    # sys.stdout.write('ddididdddddddddddddddddddddddddddddddddddddddidididid')
    # if app.clf.gpu_mode:
    #     os.system('th '+ app.config['EVAL_PATH'] + ' -model ' + app.config['GPU_MODE']  + ' -image_folder ' + app.config['IMG_PATH'] + ' -num_images 10')
    # os.system('th'+ app.config['EVAL_PATH'] + ' -model ' + app.config['CPU_MODE']  + ' -image_folder ' + app.config['IMG_PATH'] + ' -num_images 10 -gpuid -1')
    # endtime = time.time()
    # json.loggi


@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():

    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.DEBUG)

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
                sys.stdout.write(filename+'*****'+'\n')

                # Redirect the user to the uploaded_file route, which
                # will basicaly show on the browser the uploaded file
        # Load an html page with a link to each uploaded file

    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        return render_template(
            'index.html'
        )

    result  = image_caption(filenames) 
    return render_template('upload.html', filenamekks=filenames)


class ImageCaption(object):
    # 预先加载模型
    def __init__(self, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
           self.gpu_mode = True
        else:
            self.gpu_mode = False

    def classify_image(self, filename):
        try:
            params = {'ssd_anchors':self.ssd_anchors,'img_input':self.img_input,'isess':self.isess,'image_4d':self.image_4d,'predictions':self.predictions,'localisations':self.localisations,'bbox_img':self.bbox_img}

            # read image
            img = mpimg.imread(filename)


            starttime = time.time()
            rclasses, rscores, rbboxes =  process_image(img,params)
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            endtime = time.time()

            bet_result = [(str(idx+1)+' : '+ self.l_VOC_CLASS[v], '%.5f' % rscores[idx]) for idx, v in enumerate(rclasses)]


            # save image after draw box
            fileout = str(datetime.datetime.now()).replace(' ', '_') + 'processed_img' + '.jpg' 
            fileps = os.path.join(DETECTED_FOLDER, fileout)
            cv2.imwrite(fileps,img)

            new_img_base64 = embed_image_html(fileps)


            rtn = (True, (rclasses, rscores, rbboxes), bet_result, '%.3f' % (endtime - starttime))
            return rtn,new_img_base64

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


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
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    sys.stdout.write(str(opts)+' fkfkkfkf'+'\n')
    # ImageCaption.default_args.update({'gpu_mode': opts.gpu})
    app.clf = ImageCaption(opts.gpu)
    # Initialize classifier + warm start by forward for allocation
    # ckpt_path = os.getcwd() + '/checkpoints/model.ckpt'
    # init_stateModel = init_model(ckpt_path)
    # app.clf = ImagenetClassifier(**init_stateModel)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # if not os.path.exists(bk_FOLDER):
    #     os.makedirs(bk_FOLDER)
    # if not os.path.exists(DETECTED_FOLDER):
    #     os.makedirs(DETECTED_FOLDER)
    start_from_terminal(app)