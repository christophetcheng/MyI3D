// my3dviewer.cpp : Defines the entry point for the console application.
//

//#ifdef _WIN32
#include "stdafx.h"
//#endif

#ifndef _WIN32
#include <libkern/OSAtomic.h>
#include <pthread.h>
#endif

#include <GL/GLUT.h>
#include <math.h>
#include <vector>
#include <numeric>
using namespace std;
using namespace cv;
#include <opencv2/core/core.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv/highgui.h>


//GLenum rgb = GL_FALSE, doubleBuffer = GL_TRUE, directRender;

const float dests[][3] = 
{
	{ 2.5,-1.5,-1},
	{ 2.5, 0, 1},
	{ 2.5, 1.5,-1},
	{ 0, 0,-1},
	{.5,.5, 2},
	{.5,1.5,0},
	{-1, 0,.5},
	{-1,-1.5,-2},
	{ -2.,-1.5,-1},
	{ -2.5, 0, 2},
	{ -2., 1,0}
};
GLuint theTorus, theGrid;

struct FaceViewPoint { GLdouble percentx, percenty, percentz; };
const FaceViewPoint default_faceviewpoint = { .0f, .0f, .3f };
typedef const FaceViewPoint* P_FaceViewPoint;
volatile P_FaceViewPoint g_faceviewpoint = new FaceViewPoint(default_faceviewpoint);

bool use_face = false, use_tracking = false, use_camshift=true;

static const int kDepth = -4;

void
	FillTorus()
{
	float rc = 0.5;
	float rt = 0.05;

	GLUquadric* disk = gluNewQuadric();
	for(int i=0;i<5;++i)
	{
		if(i%2 == 0)
			glColor3f(.8, .2, .2);
		else
			glColor3f(.9, .9, .9);
		gluDisk(disk, i*(rc/5), (i+1)*(rc/5), 20, 20);
	}

	glColor3f(.9, .9, .9);
	GLUquadric* cyl = gluNewQuadric();
	gluQuadricDrawStyle(cyl, GLU_FILL);

	gluSphere(cyl,rt,10,10);
	glTranslatef(0, 0, kDepth);
	gluCylinder(cyl,rt,rt,-kDepth,10,10);
}

void FillGrid()
{
	int mx=-4, MX=4, my=-3, MY=3, mz=kDepth, MZ=0;

	glColor3f(.9, .9, .9);

	for( int z=mz; z<=MZ ; ++z )
	{
		glBegin(GL_LINE_LOOP);
		glVertex3f(mx, my, z);
		glVertex3f(mx, MY, z);
		glVertex3f(MX, MY, z);
		glVertex3f(MX, my, z);
		glEnd();
	}

	for( int x=mx; x<=MX ; ++x )
	{
		glBegin(GL_LINE_STRIP);
		glVertex3f(x, my, MZ);
		glVertex3f(x, my, mz);
		glVertex3f(x, MY, mz);
		glVertex3f(x, MY, MZ);
		glEnd();
	}

	for( int y=my; y<=MY; ++y )
	{
		glBegin(GL_LINE_STRIP);
		glVertex3f(mx, y, MZ);
		glVertex3f(mx, y, mz);
		glVertex3f(MX, y, mz);
		glVertex3f(MX, y, MZ);
		glEnd();
	}
}

void  Idle(void)
{
	glutPostRedisplay();
}


void
	DrawScene(void)
{
	int i;

	glPushMatrix();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	const FaceViewPoint vp = *g_faceviewpoint; // take a copy to prevent modification

	float eyedistance = 20.f - 9.f * vp.percentz; // near plane at 10.
	// the angle of the camera is 66 deg. (we remove 6 deg each side) so when percentx=100%, eyex = tan((66-2*6)/2)*eyedistance
	float eyex = vp.percentx * 0.51 * eyedistance ;
	// vertically, angle is 55 deg
	float eyey = vp.percenty * 0.38 * eyedistance;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum( (-eyex - 4) * 10/eyedistance , (-eyex + 4) * 10/eyedistance,
		(-eyey - 3) * 10/eyedistance , (-eyey + 3) * 10/eyedistance ,  10, 100);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eyex,eyey,eyedistance,eyex,eyey,0,0,1,0);

	glColor3f(.9, .9, .9);
	glCallList(theGrid);

	for (i = 0; i < sizeof(dests)/sizeof(float)/3; i++) {
		glPushMatrix();
		glTranslatef(dests[i][0] , dests[i][1] , dests[i][2]);
		glCallList(theTorus);
		glPopMatrix();
	}

	glPopMatrix();
	glutSwapBuffers();
}

cv::Point2f barycentre(const vector<cv::Point>& v)
{
	return accumulate(v.begin(), v.end(), cv::Point()) * (1.f/v.size());
}

bool findface(cv::CascadeClassifier& cascade_face, cv::Mat& img, cv::Rect& face)
{
	float searchfactor = .5f; // means we double each time the size of the search rect
	cv::Rect searchrect = face;
	vector<cv::Rect> found;
	for(int iter=0; ; ++iter)
	{
		//printf("Searching face ... iter #%d\n", iter);
		// increase by search factor
		cv::Point2f tl = searchrect.tl() - cv::Point(searchrect.size()) * searchfactor;
		cv::Point2f br = searchrect.br() + cv::Point(searchrect.size()) * searchfactor;

		searchrect = cv::Rect(tl, br);
		searchrect &= cv::Rect(cv::Point(0,0), img.size());

		cv::Mat roi(img, searchrect);
		cascade_face.detectMultiScale(roi, found, 1.5, 3, 0, cv::Size(20,20), cv::Size(100,100));

		// Let's take the first one to compute view point
		// no need for a mutex, atomic exchange is good enough
		if(!found.empty())
		{
			cv::rectangle(img, searchrect, cv::Scalar(0,255,0));
			// choose biggest face
			unsigned int maxi = 0, maxsize = 0;
			for(unsigned int i=0; i< found.size(); ++i)
			{
				unsigned int size = found[i].area();
				if(size >= maxsize)
				{
					maxsize = size;
					maxi = i;
				}
			}
			face = found[maxi] + searchrect.tl();
			//printf("Found face (%d, %d, %d, %d)\n", face.x, face.y, face.width, face.height);
			return true;
		}

		if(searchrect.size() == img.size())
			return false; // nothing found

	}
	return false;
}


#ifndef _WIN32
void* thread_face_detect(void*)
#else
unsigned long thread_face_detect (void*) 
#endif
{
	cv::VideoCapture capture;

	capture.open(0);
	if(!capture.isOpened())
	{
		printf("Cannot open capture\n");
		return 0;
	};
	cv::CascadeClassifier cascade_face;

	//#define objxml "haarcascade_frontalface_default.xml"
#define objxml "haarcascade_frontalface_alt2.xml"

#ifdef _WIN32
	const char* xml = "C:\\OpenCV2.3\\data\\haarcascades\\" objxml;
#else
	const char* xml = "/usr/local/share/OpenCV/haarcascades/" objxml;
#endif
	if(!cascade_face.load(xml)) {
		printf("Cannot load face classifier\n");
		return 0;
	}
	cv::namedWindow("camera");

	cv::Mat rawimg, img, previmg;
	//cv::Mat gray, prevgray;
	std::vector<cv::Rect> found;
	vector<float> err;
	cv::RotatedRect face_rect;
	cv::Rect face, origface;
	vector<cv::Point> trackpoints, orig_trackpoints;
	cv::Point adj_center(-1,-1), orig_offset; // the offset from the mean of orig_trackpoints to the center of face_rect
	vector<uchar> trackstatus;
	uint64 lastiter_facedetect;
	int64 prev_tick=0;
	float fps=10;
	double tickfreq = cv::getTickFrequency();
	float percentx=0, percenty=0, percentz = .5f;

	// For camshift
	bool track_object=false; // means we have found a face a initialized the histogram
    Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	int vmin = 150, vmax = 256, smin = 30;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Rect trackWindow;
    RotatedRect trackBox;
    createTrackbar( "Vmin", "camera", &vmin, 256, 0 );
    createTrackbar( "Vmax", "camera", &vmax, 256, 0 );
    createTrackbar( "Smin", "camera", &smin, 256, 0 );


	for(uint64 iter=0;;++iter)
	{
		capture >> rawimg;
		
		if(!rawimg.data)
		{
			printf("Cannot get image, sleeping 1s ...\n");
#ifdef _WIN32
			Sleep(1000);
#else
			sleep(1);
#endif
			continue;
		}
		// Display FPS
		int64 new_tick = cv::getTickCount();
		fps = .8f * fps + .2f * (tickfreq/(new_tick - prev_tick));
		prev_tick = new_tick;
		if(iter%100 == 0 )
			printf("FPS = %f\n", fps);

		if(use_face)
		{
			//img = rawimg;
			cv::resize(rawimg,img,cv::Size(), 0.5f, 0.5f);
			cv::cvtColor(img,img,CV_BGR2GRAY);
			cv::equalizeHist(img, img);
			if(previmg.empty()) img.copyTo(previmg);

			if(face.size().area() == 0)
				face = cv::Rect(cv::Point(0,0), img.size());
			if( findface(cascade_face, img, face) )
			{
				adj_center = (face.tl() + face.br()) * .5;
			}
		}
		else if(use_camshift)
		{
			cv::cvtColor(rawimg,hsv,CV_BGR2HSV);
            int _vmin = vmin, _vmax = vmax;

            inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                    Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            mixChannels(&hsv, 1, &hue, 1, ch, 1);

			// if camshift not initialized, try to find a face
			// if camshit initialized, try to re-find a face every 100 frames
			if( !track_object || iter%500 == 0)
			{
				cv::cvtColor(rawimg,img,CV_BGR2GRAY);
				cv::equalizeHist(img, img);
				face = cv::Rect(cv::Point(0,0), img.size());
				if( findface(cascade_face, img, face) )
				{
					adj_center = (face.tl() + face.br()) * .5;
					track_object = true;

					printf("found face x=%d, y=%d\n",adj_center.x, adj_center.y);


					// the following from camshitdemo: init histogram
					Mat roi(hue, face);
					Mat maskroi(mask, face);
					cv::imshow("roi",roi);
					cv::imshow("maskroi",maskroi);

					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					normalize(hist, hist, 0, 255, CV_MINMAX);
                    
					trackWindow = face;
					rectangle( rawimg, face , Scalar(0,255,0), 3, CV_AA );

					histimg = Scalar::all(0);
					int binW = histimg.cols / hsize;
					Mat buf(1, hsize, CV_8UC3);
					for( int i = 0; i < hsize; i++ )
						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
					cvtColor(buf, buf, CV_HSV2BGR);
                        
					for( int i = 0; i < hsize; i++ )
					{
						int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
						rectangle( histimg, Point(i*binW,histimg.rows),
									Point((i+1)*binW,histimg.rows - val),
									Scalar(buf.at<Vec3b>(i)), -1, 8 );
					}
				}// face found
			} // face need to be found or re-found

			if(track_object)
			{
				imshow("hue",hue);
				// The following also from camshiftdemo
			    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
				trackWindow = trackBox.boundingRect();
				adj_center = trackBox.center;

				if( trackWindow.area() <= 1 || trackBox.boundingRect().area()<=1 || (adj_center.x==0 && adj_center.y==0))
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
					track_object = false;
                }

				ellipse( rawimg, trackBox, Scalar(0,0,255), 3, CV_AA );
				printf("tracked face x=%d, y=%d\n",adj_center.x, adj_center.y);
				face = trackBox.boundingRect();
			}
			

		}
		else if(use_tracking)
		{
			// The algorithm is:
			// if we have less than 5 points to track, need face detection, otherwise use track point.
			// do face detect max every x iter.
			if(trackpoints.size() < 10 || (iter > lastiter_facedetect+30) )
			{
				lastiter_facedetect = iter;

				if( findface(cascade_face, img, face) )
				{
					face_rect = cv::RotatedRect((face.tl() + face.br()) * .5,face.size() ,0);

					printf("Detecting track points...\n");
					cv::Mat mask(img.size(),CV_8UC1,cv::Scalar(0));
					cv::ellipse(mask, face_rect, cv::Scalar(255),-1);

					cv::goodFeaturesToTrack(img, trackpoints, 20, 0.1, 10, mask);
					orig_trackpoints = trackpoints;
					printf("found %lu track points\n", trackpoints.size() );
					cv::Point c = (face.tl() + face.br()) * .5;
					cv::Point b = barycentre(orig_trackpoints);
					orig_offset = c - b;
				}
			}

			// Now compute the new location of points
			if( !trackpoints.empty() && lastiter_facedetect!=iter)
			{
				vector<cv::Point> old_trackpoints = trackpoints;
				cv::calcOpticalFlowPyrLK(previmg, img, old_trackpoints, trackpoints, trackstatus, err, cv::Size(10,10), 2);
				std::vector<cv::Point>::iterator i = trackpoints.begin(),
					k = old_trackpoints.begin(),
					l = orig_trackpoints.begin();

				for(std::vector<unsigned char>::iterator j=trackstatus.begin(); j!=trackstatus.end(); ++j)
				{
					if( ! *j ) {
						i = trackpoints.erase(i);
						l = orig_trackpoints.erase(l); // the bug is here
						k = old_trackpoints.erase(k);
						// recompute offset
						cv::Point c = (face.tl() + face.br()) * .5;
						cv::Point b = barycentre(orig_trackpoints);
						orig_offset = c - b;
						printf("Lost a trackpoint - now %lul\n", trackpoints.size());
					}
					else {
						cv::line(img, *i, *k, cv::Scalar(0,255,0));
						cv::circle( img, *i, 3, cv::Scalar(0,255,0));
						cv::circle( img, *l, 3, cv::Scalar(255,0,0));
						++i, ++k, ++l;
					}
				}
			}

			// Now compute the mean of track points.
			if( !trackpoints.empty() )
			{
				cv::Point center = barycentre(trackpoints);
				cv::Point adj_center = center  + orig_offset;

				cv::circle(img, center, 2, cv::Scalar(0, 0, 255));
				cv::line(img, center, adj_center, cv::Scalar(255,0,0));

				// we can remove points to far off the adjusted center for the next loop
				float maxdist =  cv::norm( cv::Point2f(face.width/2., face.height/2) );
				for( std::vector<cv::Point>::iterator i = trackpoints.begin(); i != trackpoints.end(); )
				{
					float dist = cv::norm( *i - adj_center );
					if( dist > maxdist ) {
						i = trackpoints.erase(i);
						printf("cleaned up a trackpoint\n");
					}
					else {
						++i;
					}
				}
			}
		}


		if(adj_center.x >= 0)
		{
			float newpercentx = 2 * (1.f * adj_center.x / rawimg.size().width) - 1;
			float newpercenty = 2 * (1.f * adj_center.y / rawimg.size().height) - 1;
			// the ratio 3.f is based on the fact that the closest face detection gives a face area 2 times smalles than img area
			float newpercentz = (2.f * face.area()) / rawimg.size().area();
			if(newpercentz>1.f) newpercentz=1.f;

			// now do a smooth move to the new location
			const float smooth_factor = .7f;
			percentx = (1.f - smooth_factor) * percentx + smooth_factor * newpercentx;
			percenty = (1.f - smooth_factor) * percenty + smooth_factor * newpercenty;
			percentz = (1.f - smooth_factor) * percentz + smooth_factor * newpercentz;

			printf("\t\t percent x,y,z = %f,%f,%f\n", percentx, percenty, percentz);

			FaceViewPoint* newpoint = new FaceViewPoint(default_faceviewpoint);
			newpoint->percentx = -percentx ;
			newpoint->percenty = -percenty ;
			newpoint->percentz = percentz;

			// spin to swap value
			const FaceViewPoint* oldvalue;
#ifdef _WIN32
			PVOID output = InterlockedExchangePointer((PVOID*)&g_faceviewpoint, newpoint);
			oldvalue = static_cast<const FaceViewPoint*>(output);
#else
			while(1)
			{
				oldvalue = g_faceviewpoint;
				if(OSAtomicCompareAndSwapPtr((void*)oldvalue, (void*)newpoint, (void**)&g_faceviewpoint))
				{
					break;
				}
			}
#endif
			delete oldvalue;
		}

		// Finally store prevgray
		img.copyTo(previmg);

		cv::ellipse(img, face_rect, cv::Scalar(255,0,0));
		cv::rectangle(img, face, cv::Scalar(255,0,0));

		cv::imshow("camera",rawimg);
		int c = cv::waitKey(1)&0xFF;
		c = tolower(c);
		// End processing on ESC, q or Q
		if(c == 27 || c == 'q')
			break;

	}

	cv::destroyAllWindows();
	capture.release();
	exit(0);
	return 0;
}

void
	Init(void)
{

#ifndef _WIN32
	pthread_t thread;
	pthread_create(&thread, 0, thread_face_detect, 0);
#else
	DWORD threadID;
	CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)thread_face_detect, 0, 0, &threadID);
#endif

	theTorus = glGenLists(1);
	glNewList(theTorus, GL_COMPILE);
	FillTorus();
	glEndList();

	theGrid = glGenLists(1);
	glNewList(theGrid, GL_COMPILE);
	FillGrid();
	glEndList();

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);

	glClearIndex(8);
	glShadeModel(GL_FLAT);

	glMatrixMode(GL_PROJECTION);
	gluPerspective(45, 1.33, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);

}

void
	Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
}

/* ARGSUSED1 */
void
	Key(unsigned char key, int x, int y)
{

	switch (key) {
	case 27:
		exit(0);
		break;
	case ' ':
		glutIdleFunc(Idle);
		break;
	case 'c':
		use_camshift = true;
		use_face = use_tracking = false;
		break;
	case 'f':
		use_face = true;
		use_camshift = use_tracking = false;
		break;
	case 't':
		use_tracking = true;
		use_face = use_camshift = false;
		break;
	}
}

void
	visible(int vis)
{
	if (vis == GLUT_VISIBLE) {
		glutIdleFunc(Idle);
	} else {
		glutIdleFunc(NULL);
	}
}

int
	main(int argc, char **argv)
{ 
	glutInitWindowSize(400, 300);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("Olympic");

	Init();

	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Key);
	glutDisplayFunc(DrawScene);

	glutVisibilityFunc(visible);

	if(argc >= 2 && 0==strcmp(argv[1],"-f"))
		glutFullScreen();

	glutMainLoop();
	return 0;             /* ANSI C requires main to return int. */
}
