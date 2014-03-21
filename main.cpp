/**
 * Copyright (C) 2014 MightyCid
 *
 * This file is part of CUDA-pathtracer <github.com/mightycid/CUDA-pathtracer>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include "globals.h"
#include "pathtracer.h"
#include "cutil.h"

#include <cuda_gl_interop.h>

bool InitGL(int *argc, char** argv);
void InitPBO();
bool InitPathtracer();
void RunPathtracer();
void ReleasePathtracer();

void Display();
void Idle();
void Keyboard(unsigned char key, int /*x*/, int /*y*/);
void PressKey(int key, int /*x*/, int /*y*/);
void GetFPS();

Camera camera;
float timer = 0.0f;
uint32_t frameCount = 0, timeBase = 0;
uint32_t width, height;
GLuint pbo;
GLuint textureId;

Pathtracer *pathtracer = NULL;

bool InitGL(int argc, char** argv) {
	glutInit(&argc, argv);

	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	glutInitContextProfile(GLUT_CORE_PROFILE);

    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Pathtracer");
    glutDisplayFunc(Display);
	glutIdleFunc(Idle);
    glutKeyboardFunc(Keyboard);
	//TODO mouse camera movement
    //glutMotionFunc(motion);
	glutSpecialFunc(PressKey);

	glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

    // viewport
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	InitPBO();

    return true;
}

/**
 * Generates the pixel buffer object for direct rendering on the GPU
 */
void InitPBO() {
  uint32_t numTexels = width * height;
  uint32_t sizeTexData = sizeof(GLfloat) * numTexels * 3;
  void *data = malloc(sizeTexData*sizeof(float));
  
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glBufferData(GL_ARRAY_BUFFER, sizeTexData, data, GL_DYNAMIC_DRAW);

  CudaSafeCall(cudaGLRegisterBufferObject(pbo));

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1,&textureId);

  glBindTexture( GL_TEXTURE_2D, textureId);

  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_FLOAT, NULL);

  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
}

/**
 * Initializes the scene
 * creates camera, materials, primitives and lights
 * TODO: configuration file parsing
 */
void InitScene(Camera *camera, std::vector<Material> *mv, std::vector<Primitive> *pv, std::vector<Light> *lv) {
	*camera = Camera(Point(0,45,79.5), Point(0,35,0), Vec(0,1,0), width, height, 0.f, 0.f, 60.f);

	*mv = std::vector<Material>();
	
	mv->push_back(CreateDiffuseMaterial(Color(1.f, 1.f, 1.f), 0.f)); // default
	mv->push_back(CreateDiffuseMaterial(Color(0.75f, 0.25f, 0.25f), 0.f)); // 1
	mv->push_back(CreateDiffuseMaterial(Color(0.25f, 0.25f, 0.75f), 0.f)); // 2
	mv->push_back(CreateDiffuseMaterial(Color(0.75f, 0.75f, 0.75f), 0.f)); // 3
	mv->push_back(CreateSpecularMaterial(Color(0.999f, 0.999f, 0.999f), 1.f)); // 4
	mv->push_back(CreateTransmissiveMaterial(Color(0.999f, 0.999f, 0.999f), 1.5f)); // 5

	*pv = std::vector<Primitive>();
	//scene 1
	/*pv->push_back(Primitive(Point(0.f,-1E+5f-1.f,0.f), 1E+5f, 1)); //floor
	pv->push_back(Primitive(Point(0.f,1E+5f+3.f,0.f), 1E+5f, 1)); //ceiling
	pv->push_back(Primitive(Point(0.f,0.f,-1E+5f-7.f), 1E+5f, 1)); //back
	pv->push_back(Primitive(Point(0.f,0.f,1E+5f+7.f), 1E+5f, 1)); //front
	pv->push_back(Primitive(Point(-1E+5f-4.f,0.f,0.f), 1E+5f, 2)); //left
	pv->push_back(Primitive(Point(1E+5f+4.f,0.f,0.f), 1E+5f, 3)); //right
	pv->push_back(Primitive(Point(-1.5f,0.f,0.f), 1.f, 4));
	pv->push_back(Primitive(Point(1.5f,0.f,0.f), 1.f, 5));
	pv->push_back(Primitive(Point(0.f,2.f,0.f), 0.5f, 1, 0));*/

	//scene 2
	pv->push_back(Primitive(Point( 1e5+50,     40,      0),  1e5, 2)); //left
	pv->push_back(Primitive(Point(-1e5-50,     40,      0),  1e5, 1)); //right
	pv->push_back(Primitive(Point(      0,     40,-1e5-80),  1e5, 3)); //back
	pv->push_back(Primitive(Point(      0,     40, 1e5+80),  1e5, 3)); //front
	pv->push_back(Primitive(Point(      0,   -1e5,      0),  1e5, 3)); //bottom
	pv->push_back(Primitive(Point(      0, 1e5+80,      0),  1e5, 3)); //top
	pv->push_back(Primitive(Point(    -25,   16.5,    -50), 16.5, 4));
	pv->push_back(Primitive(Point(     25,   16.5,    -25), 16.5, 5));
	pv->push_back(Primitive(Point(      0,  579.6,    -40),  500, 1, 0));

	*lv = std::vector<Light>();
	lv->push_back(Light(8, Color(12.f,12.f,12.f)));
	//lv->push_back(Light(Point(50,20,40.f), Color(150.f,150.f,150.f)));
}

/**
 * Initialize pathtracer
 */
bool InitPathtracer() {
	std::vector<Material> materials;
	std::vector<Primitive> primitives;
	std::vector<Light> lights;
	InitScene(&camera, &materials, &primitives, &lights);
	
	pathtracer = new Pathtracer(&camera, 10);
	if(!pathtracer) {
        fprintf(stderr, "ERROR: Could not create Pathtracer instance.");
        fflush(stderr);
        return false;
	}
	if(!pathtracer->Init(materials, primitives, lights)) {
        fprintf(stderr, "ERROR: Could not initialize Pathtracer.");
        fflush(stderr);
        return false;
	}

	return true;
}

/**
 * Release resources
 */
void ReleasePathtracer() {
	pathtracer->Release();
	delete pathtracer;
}

/**
 * default display method
 * maps the pbo to the device buffer and calls the kernel before displaying
 * the result as a texture in OpenGL
 */
void Display() {
	//Clear the color buffer
	glClear(GL_COLOR_BUFFER_BIT);
	
	if(camera.IsUpdated()) pathtracer->Reset();

	float* devBuffer = NULL;
	CudaSafeCall(cudaGLMapBufferObject((void**)&devBuffer, pbo));

	// call kernel
	pathtracer->Run(devBuffer);
	
	CudaSafeCall(cudaGLUnmapBufferObject(pbo));

	// render result as texure
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, NULL);

	glBegin(GL_QUADS); {
		glTexCoord2f(0, 0); glVertex3f(-1, 1, 0);
		glTexCoord2f(1, 0); glVertex3f(1, 1, 0);
		glTexCoord2f(1, 1); glVertex3f(1, -1, 0);
		glTexCoord2f(0, 1); glVertex3f(-1, -1, 0);
	} glEnd();

	glutSwapBuffers();
	GetFPS();
}

void Idle() {
	glutPostRedisplay();
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
        case (27) :
            exit(EXIT_SUCCESS);
            break;
        case 'r' :
            exit(EXIT_SUCCESS);
            break;
	}
}

/**
 * TODO camera movement through keyboard commands
 */
void PressKey(int key, int /*x*/, int /*y*/) {
	if(glutGetModifiers()==GLUT_ACTIVE_CTRL) {
		switch(key) {
			case GLUT_KEY_UP:
				camera.Translate(Vec(0.f, 0.f, CAM_TRANSLATE_DELTA));
				break;
			case GLUT_KEY_RIGHT:
				camera.Translate(Vec(CAM_TRANSLATE_DELTA, 0.f, 0.f));
				break;
			case GLUT_KEY_DOWN:
				camera.Translate(Vec(0.f, 0.f, -CAM_TRANSLATE_DELTA));
				break;
			case GLUT_KEY_LEFT:
				camera.Translate(Vec(-CAM_TRANSLATE_DELTA, 0.f, 0.f));
				break;
		}
	} else {
		switch(key) {
			case GLUT_KEY_UP:
				camera.Rotate(Vec(0.f, CAM_ROTATE_DELTA, 0.f));
				break;
			case GLUT_KEY_RIGHT:
				camera.Rotate(Vec(-CAM_ROTATE_DELTA, 0.f, 0.f));
				break;
			case GLUT_KEY_DOWN:
				camera.Rotate(Vec(0.f, -CAM_ROTATE_DELTA, 0.f));
				break;
			case GLUT_KEY_LEFT:
				camera.Rotate(Vec(CAM_ROTATE_DELTA, 0.f, 0.f));
				break;
		}
}
}

void GetFPS() {
	++frameCount;
	uint32_t currTime = glutGet(GLUT_ELAPSED_TIME);

	uint32_t elapsed = currTime - timeBase;
	if (elapsed > 1000) {
		float fps = frameCount*1000.0f/elapsed;
		timeBase = currTime;
		frameCount = 0;

		char buffer[32];
		sprintf(buffer, "Pathtracer (%.4f sps : %u)", fps, pathtracer->GetIteration());
		glutSetWindowTitle(buffer);
	}
}


/**
 * Main method
 */
int main(int argc, char** argv) {
	Scene *scene = NULL;
	width = 640;
	height = 480;
	printf("Initializing OpenGL...");
	if(!InitGL(argc, argv)) {
        fprintf(stderr, "ERROR: Could not initialize Pathtracer.");
        fflush(stderr);
		exit(-1);
	}
	printf("done!\n");

	printf("Initializing Pathtracer...");
	if(!InitPathtracer()) {
        fprintf(stderr, "ERROR: Could not initialize Pathtracer.");
        fflush(stderr);
		exit(-1);
	}
	printf("done!\n");

	printf("Starting main loop\n");
	glutMainLoop();
	ReleasePathtracer();
}