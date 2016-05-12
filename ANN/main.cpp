/*�ο���http://blog.csdn.net/jinshengtao/article/details/17954427*/

#include <cv.h>  
#include <highgui.h>  
#include <cvaux.h>  
#include <ml.h>    
#include <iostream>  
#include <vector>  
#include "Plate.h"  

#define HORIZONTAL    1  
#define VERTICAL    0  
using namespace std;  
using namespace cv;

CvANN_MLP  ann;  
const char strCharacters[] = {'0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};  
const int numCharacters=30; 

  
//��minAreaRect��õ���С��Ӿ��Σ����ݺ�Ƚ����ж�  
bool verifySizes(RotatedRect mr)  
{  
    float error=0.4;  
    //Spain car plate size: 52x11 aspect 4,7272  
    float aspect=4.7272;  
    //Set a min and max area. All other patchs are discarded  
    int min= 15*aspect*15; // minimum area  
    int max= 125*aspect*125; // maximum area  
    //Get only patchs that match to a respect ratio.  
    float rmin= aspect-aspect*error;  
    float rmax= aspect+aspect*error;  
  
    int area= mr.size.height * mr.size.width;  
    float r= (float)mr.size.width / (float)mr.size.height;  
    if(r<1)  
        r= (float)mr.size.height / (float)mr.size.width;  
  
    if(( area < min || area > max ) || ( r < rmin || r > rmax )){  
        return false;  
    }else{  
        return true;  
    }  
  
}  
  
//ֱ��ͼ���⻯  
Mat histeq(Mat in)  
{  
    Mat out(in.size(), in.type());  
    if(in.channels()==3){  
        Mat hsv;  
        vector<Mat> hsvSplit;  
        cvtColor(in, hsv, CV_BGR2HSV);  
        split(hsv, hsvSplit);  
        equalizeHist(hsvSplit[2], hsvSplit[2]);  
        merge(hsvSplit, hsv);  
        cvtColor(hsv, out, CV_HSV2BGR);  
    }else if(in.channels()==1){  
        equalizeHist(in, out);  
    }  
  
    return out;  
  
}  
  
Mat process1(char *fileRead, char *fileWrite) 
{  
    Mat img_gray = imread(fileRead,CV_LOAD_IMAGE_GRAYSCALE);  
    Mat input = imread(fileRead);  
    //char res[20];  
  
    //apply a Gaussian blur of 5 x 5 and remove noise  
    blur(img_gray,img_gray,Size(5,5));  
  
    //Finde vertical edges. Car plates have high density of vertical lines  
    Mat img_sobel;  
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);//xorder=1,yorder=0,kernelsize=3  
  
    //apply a threshold filter to obtain a binary image through Otsu's method  
    Mat img_threshold;  
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);  
  
    //Morphplogic operation close:remove blank spaces and connect all regions that have a high number of edges  
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3) );  
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);  
  
     //Find ���� of possibles plates  
    vector< vector< Point> > contours;  
    findContours(img_threshold,  
        contours, // a vector of contours  
        CV_RETR_EXTERNAL, // ��ȡ�ⲿ����  
        CV_CHAIN_APPROX_NONE); // all pixels of each contours  
  
    //Start to iterate to each contour founded  
    vector<vector<Point> >::iterator itc= contours.begin();  
    vector<RotatedRect> rects;  
  
    //Remove patch that are no inside limits of aspect ratio and area.      
    while (itc!=contours.end()) {  
        //Create bounding rect of object  
        RotatedRect mr= minAreaRect(Mat(*itc));  
        if( !verifySizes(mr)){  
            itc= contours.erase(itc);  
        }else{  
            ++itc;  
            rects.push_back(mr);  
        }  
    }  
  
    // Draw blue contours on a white image  
    cv::Mat result;  
    //input.copyTo(result);  
    //cv::drawContours(result,contours,  
    //  -1, // draw all contours  
    //  cv::Scalar(0,0,255), // in blue  
    //  3); // with a thickness of 1  
  
  
	Mat grayResult;  

    for(int i=0; i< rects.size(); i++)  
    {  
        //For better rect cropping for each posible box  
        //Make floodfill algorithm because the plate has white background  
        //And then we can retrieve more clearly the contour box  
        circle(result, rects[i].center, 3, Scalar(0,255,0), -1);  
        //get the min size between width and height  
        float minSize=(rects[i].size.width < rects[i].size.height)?rects[i].size.width:rects[i].size.height;  
        minSize=minSize-minSize*0.5;  
        //initialize rand and get 5 points around center for floodfill algorithm  
        srand ( time(NULL) );  
        //Initialize floodfill parameters and variables  
        Mat mask;  
        mask.create(input.rows + 2, input.cols + 2, CV_8UC1);  
        mask= Scalar::all(0);  
        int loDiff = 30;  
        int upDiff = 30;  
        int connectivity = 4;  
        int newMaskVal = 255;  
        int NumSeeds = 10;  
        Rect ccomp;  
        int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;  
        for(int j=0; j<NumSeeds; j++){  
            Point seed;  
            seed.x=rects[i].center.x+rand()%(int)minSize-(minSize/2);  
            seed.y=rects[i].center.y+rand()%(int)minSize-(minSize/2);  
            circle(result, seed, 1, Scalar(0,255,255), -1);  
            int area = floodFill(input, mask, seed, Scalar(255,0,0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);  
        }  
        //sprintf(res,"result%d.jpg",i);  
        //imwrite(res,mask);  
  
        //Check new floodfill mask match for a correct patch.  
        //Get all points detected for get Minimal rotated Rect  
        vector<Point> pointsInterest;  
        Mat_<uchar>::iterator itMask= mask.begin<uchar>();  
        Mat_<uchar>::iterator end= mask.end<uchar>();  
        for( ; itMask!=end; ++itMask)  
            if(*itMask==255)  
                pointsInterest.push_back(itMask.pos());  
  
        RotatedRect minRect = minAreaRect(pointsInterest);  
  
        if(verifySizes(minRect)){  
            // rotated rectangle drawing   
            Point2f rect_points[4]; minRect.points( rect_points );  
            for( int j = 0; j < 4; j++ )  
                line( result, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255), 1, 8 );      
  
            //Get rotation matrix  
            float r= (float)minRect.size.width / (float)minRect.size.height;  
            float angle=minRect.angle;      
            if(r<1)  
                angle=90+angle;  
            Mat rotmat= getRotationMatrix2D(minRect.center, angle,1);  
  
            //Create and rotate image  
            Mat img_rotated;  
            warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);  
  
            //Crop image  
            Size rect_size=minRect.size;  
            if(r < 1)  
                swap(rect_size.width, rect_size.height);  
            Mat img_crop;  
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);  
  
            Mat resultResized;  
            resultResized.create(33,144, CV_8UC3);  
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);  
            //Equalize croped image  
            //Mat grayResult;  
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);   
            blur(grayResult, grayResult, Size(3,3));  
            grayResult=histeq(grayResult);  
        /*  if(1){  
                stringstream ss(stringstream::in | stringstream::out); 
                ss << "haha" << "_" << i << ".jpg"; 
                imwrite(ss.str(), grayResult); 
            }*/  
            //output.push_back(Plate(grayResult,minRect.boundingRect()));  
        }  
    }  

    //imshow("car_plate",grayResult);  
	imwrite(fileWrite, grayResult);
    waitKey(0);  


    return grayResult;  
}




//����Ӧ��������֤ ��ס���ǲ���һ�����֡������Ǹ��ݳ��������
bool verifySizes(Mat r){  
    //Char sizes 45x77  
    float aspect=45.0f/77.0f;  
    float charAspect= (float)r.cols/(float)r.rows;  
    float error=0.35;  
    float minHeight=15;  
    float maxHeight=28;  
    //We have a different aspect ratio for number 1, and it can be ~0.2  
    float minAspect=0.2;  
    float maxAspect=aspect+aspect*error;  
    //area of pixels  
    float area=countNonZero(r);  
    //bb area  
    float bbArea=r.cols*r.rows;  
    //% of pixel in area  
    float percPixels=area/bbArea;  
  
    /*if(DEBUG) 
    cout << "Aspect: "<< aspect << " ["<< minAspect << "," << maxAspect << "] "  << "Area "<< percPixels <<" Char aspect " << charAspect  << " Height char "<< r.rows << "\n";*/  
    if(percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && r.rows >= minHeight && r.rows < maxHeight)  
        return true;  
    else  
        return false;  
  
}  
 
//������Ҫ�ǽ�����ת�任 �� �ߴ������
Mat preprocessChar(Mat in){  
    //Remap image  
    int h=in.rows;  
    int w=in.cols;  
    int charSize=20;    //ͳһÿ���ַ��Ĵ�С  

    Mat transformMat=Mat::eye(2,3,CV_32F); 

    int m=max(w,h);  
    transformMat.at<float>(0,2)=m/2 - w/2;  
    transformMat.at<float>(1,2)=m/2 - h/2;  
  
    Mat warpImage(m,m, in.type());   //������Ҫ�ǽ�����ת�任 �� �ߴ������
    warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0) );  
  
    Mat out;  
    resize(warpImage, out, Size(charSize, charSize) );   
  
    return out;  
}  
 
//create the accumulation histograms,img is a binary image, t is ˮƽ��ֱ  
Mat ProjectedHistogram(Mat img, int t)  
{  
    int sz=(t)?img.rows:img.cols;  
    Mat mhist=Mat::zeros(1,sz,CV_32F);  
  
    for(int j=0; j<sz; j++){  
        Mat data=(t)?img.row(j):img.col(j);  
        mhist.at<float>(j)=countNonZero(data);    //ͳ����һ�л�һ���У�����Ԫ�صĸ����������浽mhist��  
    }  
  
    //Normalize histogram  
    double min, max;  
    minMaxLoc(mhist, &min, &max);  
  
    if(max>0)  
        mhist.convertTo(mhist,-1 , 1.0f/max, 0);//��mhistֱ��ͼ�е����ֵ����һ��ֱ��ͼ  
  
    return mhist;  
}  
 

Mat features(Mat in, int sizeData){  
    //Histogram features  
    Mat vhist=ProjectedHistogram(in,VERTICAL);  
    Mat hhist=ProjectedHistogram(in,HORIZONTAL);  
  
    //Low data feature  
    Mat lowData;  
    resize(in, lowData, Size(sizeData, sizeData) );  
  
    //Last 10 is the number of moments components  
    int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;  
  
    Mat out=Mat::zeros(1,numCols,CV_32F);  
    //Asign values to feature,ANN����������Ϊˮƽ����ֱֱ��ͼ�͵ͷֱ���ͼ������ɵ�ʸ��  
    int j=0;  
    for(int i=0; i<vhist.cols; i++)  
    {  
        out.at<float>(j)=vhist.at<float>(i);  
        j++;  
    }  
    for(int i=0; i<hhist.cols; i++)  
    {  
        out.at<float>(j)=hhist.at<float>(i);  
        j++;  
    }  
    for(int x=0; x<lowData.cols; x++)  
    {  
        for(int y=0; y<lowData.rows; y++){  
            out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);  
            j++;  
        }  
    }  
      
    return out;  
}  
  
//����ǽ���Ԥ��ĺ�����������Ƕ���õ�������Ԥ��Ҳ�Ƿ�װ�ں����н����ˡ�������ǲ��Ǻ�ѵ������һ���ģ�������
int classify(Mat f){  
    
	
	int result=-1;  
    Mat output(1, 30, CV_32FC1); //����������ֻ��30���ַ� ��1��30�е�Mat ����CV_32FC1 ��ʾÿ��Ԫ�ض���float����

    ann.predict(f, output);		 //�������Ԥ�⡣����� 1��30�е�Mat �����С�ANN �Դ��ģ����ù��ġ�

    Point maxLoc;  
    double maxVal;  
    minMaxLoc(output, 0, &maxVal, 0, &maxLoc);  //Ѱ�����ֵ����Сֵ��λ�á�0 ��λ�ñ�ʾ�ò�������Ҫ��

    //We need know where in output is the max val, the x (cols) is the class.   
    return maxLoc.x;							//x�����ڵڼ��е����ꡣ����Ҫע�� С�ĳ���
}  
    

//ѵ���Ĺ��̡���Ҫ����ANN��ѵ��������װ����������ˡ�
//TrainData ��ȫ��ѵ��������ɵľ���ÿһ����һ��������������ӳÿһ��������ȡ����������Ŀ��
//classes ��ȫ����������ǩ ��TrainData ������ͬ��ֻ��һ���С�
//nlayers ����������Ԫ����Ŀ��
void train(Mat TrainData, Mat classes, int nlayers){  

    Mat layers(1,3,CV_32SC1);  

    layers.at<int>(0)= TrainData.cols;  // �����Ϊ675�У�����������ô���õ��У�������������������
    layers.at<int>(1)= nlayers;  
    layers.at<int>(2)= 30;			   //��Ϊ��30���ַ�

	//int i = TrainData.cols;
	//int j = nlayers;
	//Mat layerSizes=(Mat_<int>(1,5) << 5,2,2,2,5); 
	//cout << layerSizes << endl;
	//ann.create(layerSizes,CvANN_MLP::SIGMOID_SYM);

    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);  
  
    //Prepare trainClases  
    //Create a mat with n trained data by m classes

	//������������������675�к�30�С�675�з�ӳ�ܹ�675��������30�б�ʾ������30�ַ��ࡣ
	//
    Mat trainClasses;  
    trainClasses.create( TrainData.rows, 30, CV_32FC1 );     
	
	//���������������������0 �� 1 ����ˡ�ÿһ�е�1���ڵ��б�ʾ ���������ı�ǩ��
    for( int i = 0; i <  trainClasses.rows; i++ )  
    {  
        for( int k = 0; k < trainClasses.cols; k++ )  
        {  
            //If class of data i is same than a k class  
            if( k == classes.at<int>(i) )  
                trainClasses.at<float>(i,k) = 1;  
            else  
                trainClasses.at<float>(i,k) = 0;  
        }  
    } 

	//1 ��n�еľ���Mat��n ����ѵ�����������������ȫ����1��䡣
    Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );  
  
    //Learn classifier  
	//������TrainData��675*n��trainClasses��675*30��weight��1*n��
    ann.train( TrainData, trainClasses, weights );  
}  
  

int main()
{  
	
	char *fileRead = "C:\\data\\chepai.png";
	char *fileWrite = "C:\\data\\chepai_2.png";

	//�������Ӧ�����Ƚ����ƿٳ����������������������ࡣ
	process1(fileRead,fileWrite);


	//�����������粿��
    Mat input = imread(fileWrite,CV_LOAD_IMAGE_GRAYSCALE);  
    

	Plate mplate;  

    //Read file storage.  
    FileStorage fs;  
    fs.open("C:\\data\\OCR.xml", FileStorage::READ);		//������ļ�����ǰ����õģ���������������ȡ������ɵ�һ������

    Mat TrainingData;  
    Mat Classes;  

	//�������Խ����ݱ�ǩ���Ӧ������д���������Mat
    fs["TrainingDataF15"] >> TrainingData;		//��ѵ������ɵ�������ȡ����
    fs["classes"] >> Classes;					//ѵ����������Ӧ�ı�ǩ���Ǹ�һά�о���
	

    //ѵ�������硣������һ��������ģ��  
    train(TrainingData, Classes, 10);				//������ѵ���Ĺ��̣�����Ҫ����ѵ����װ��һ�������н��С�
	//ann.save("C:\\data\\20140105142359500.xml");	//��ѵ��������棬�´ο���ֱ����
	ann.load("C:\\data\\20140105142359500.xml");    //Ҳ����ֱ�Ӽ���֮ǰѵ���õ�ģ�͡�


	//�������Ԥ��Ĺ���
    //dealing image and save each character image into vector<CharSegment>  
    //Threshold input image      
	Mat img_threshold;  
    threshold(input, img_threshold, 60, 255, CV_THRESH_BINARY_INV);  
  
    Mat img_contours;  
    img_threshold.copyTo(img_contours);  


    //Find contours of possibles characters  ��������Ѱ��������

    vector< vector< Point> > contours;  
    findContours(img_contours,  
        contours, // a vector of contours  
        CV_RETR_EXTERNAL, // retrieve the external contours  
        CV_CHAIN_APPROX_NONE); // all pixels of each contours  

	//Start to iterate to each contour founded  
    vector<vector<Point> >::iterator itc= contours.begin();    
    //Remove patch that are no inside limits of aspect ratio and area.      
    while (itc!=contours.end()) {  
  
        //Create bounding rect of object  
        Rect mr= boundingRect(Mat(*itc));     //Ѱ��ÿһ������ס�ľ��Ρ�
        //rectangle(result, mr, Scalar(255,0,0),2);  
        //Crop image  
        Mat auxRoi(img_threshold, mr);		//��������ã�ֻ�ǽ���ס�Ĳ��� ���Ƶ���һ��Mat ��ȥ�ˡ�
        
		if(verifySizes(auxRoi)){			//if �����н������㳤������Ķ������˵��ˡ�


            auxRoi=preprocessChar(auxRoi);   //��һ������ʲôԤ����  //������Ҫ�ǽ�����ת�任 �� �ߴ������
  
            //��ÿһ��С���飬��ȡ�����ȫ������  
            Mat f=features(auxRoi,15);			//���Mat f ��һ��1ά����������������ͼƬҪ��ȡ��ȫ����������������

            //For each segment feature Classify  Ԥ��Ҳ�Ƿ�װ�ں����н�����
            int character=classify(f);         //�����ǽ���Ԥ�⡣��ÿһ���������������ȥ�� int ���ص���ʵ�Ǹ������������

            mplate.chars.push_back(strCharacters[character]);  
            mplate.charsPos.push_back(mr);  //����һ��һֱ��ִ�У����ڶ����ڲ���Ա����Ϊ0�����������vector Խ�����
            
        }  
        ++itc;  
    } 


    string licensePlate=mplate.str();  
    cout<<licensePlate<<endl;  
      
	
	system("pause");
    return 0;  
}  