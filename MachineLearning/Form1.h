#pragma once
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
//#include <random>
#define MaxNumClusters 255
struct pData {
	double X;
	double Y;
	int ClassKind;
};
namespace MachineLearning {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Drawing::Imaging;
	using namespace System::Runtime::InteropServices;
	using namespace std;


	/// <summary>
	/// Form1 ªººK­n
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:

		Bitmap^ myBitmap;
		Graphics^ g;
		Brush^ bshDraw;
		Pen^ penDraw;
		String^ Filename1;
		unsigned char PointSize, PointSize1, PointSize2, Distribution;
		pData* InputData;
		double Pi, CenterX, CenterY;
		int ClassKind, MethodCodeValue, NumberOfData, NumberOfPoint, MaxSizeOfData;
		int imW, imH, X_Cur, Y_Cur, RangeX, RangeY, NumOfCluster, NumOfClass, NumClass1, NumClass2, NumNoclass;
		bool HandFlag;
		//Bayesian Parameters
		double MeanX1, MeanY1, Sigma2X1, Sigma2Y1, SigmaX1, SigmaY1, SigmaXY1, detA1, Correlation1, Correlation12, PClass1, *PxyClass1;
		double MeanX2, MeanY2, Sigma2X2, Sigma2Y2, SigmaX2, SigmaY2, SigmaXY2, detA2, Correlation2, Correlation22, PClass2, *PxyClass2;
		int NLdegree; //polynomial degree

		//clustering, K-means, FCM, EM used
		unsigned char NumOfClusters;
		int * BackupClassKind;
		bool STOPFlag;
		unsigned char * InputDataClusterType;
		pData * ClusterCenter;
		unsigned short * Radius;
		double ** dist;
		double ** uij;

		//K-NN Classification
		unsigned char kNNs;
		bool CreatekNNFlag; //built kNNtable flag
		int totalCTestData, MaxKNN;
		short **ALLNNs, *ALLCountClass1, *ALLCountClass2;

		//K-NN Regression
		double **BDdist; //distance between test and input Data
		int totalRTestData, **NNs;
		bool BuiltkNNFlag; //built kNNtable flag

		//Perceptron, LVQ
		pData *W;
		double Bias;

		int tF;

		
	private: System::Windows::Forms::ToolStripMenuItem^  openToolStripMenuItem;
	public: 
	private: System::Windows::Forms::ToolStripMenuItem^  saveToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveAsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  exitToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  clearImageToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showDataToolStripMenuItem;
	private: System::Windows::Forms::GroupBox^  groupBox8;
	private: System::Windows::Forms::ComboBox^  comboBox_Run;

	private: System::Windows::Forms::GroupBox^  groupBox9;
	private: System::Windows::Forms::ComboBox^  comboBox_classify;



	private: System::Windows::Forms::GroupBox^  groupBox10;
	private: System::Windows::Forms::Label^  label14;
	private: System::Windows::Forms::ComboBox^  comboBox_clusters;

	private: System::Windows::Forms::ComboBox^  comboBox_clustering;

	private: System::Windows::Forms::Label^  label15;
	private: System::Windows::Forms::GroupBox^  groupBox11;
	private: System::Windows::Forms::ComboBox^  comboBox_NL_degree;

	private: System::Windows::Forms::ComboBox^  comboBox_regression;

	private: System::Windows::Forms::GroupBox^  groupBox12;
	private: System::Windows::Forms::GroupBox^  groupBox13;
	private: System::Windows::Forms::TextBox^  textBox_MaxIter;

	private: System::Windows::Forms::Label^  label17;
	private: System::Windows::Forms::TextBox^  textBox_delta;

	private: System::Windows::Forms::Label^  label16;


	private: System::Windows::Forms::ToolStripMenuItem^  showResultToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showContourToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  showMeansToolStripMenuItem;
	private: System::Windows::Forms::GroupBox^  groupBox14;
	private: System::Windows::Forms::TextBox^  textBox3;
	private: System::Windows::Forms::Label^  label18;
	private: System::Windows::Forms::CheckBox^  checkBox_Unbiased;
	private: System::Windows::Forms::ToolStripMenuItem^  showRegressionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  newFileToolStripMenuItem;

	public:
		//Regression--Linear
		double LR_a1, LR_a0;
		//Regression--Nonlinear
		double **A, *B, *NLcoef; //AX=B ==> solve equation a0+a1X+...+adX^dfor (a0,a1,...,ad)=NLcoef[]
	private: System::Windows::Forms::ToolStripMenuItem^  showClusteredToolStripMenuItem;
	public: 
	private: System::Windows::Forms::ToolStripMenuItem^  showClusterCenterToolStripMenuItem;
	private: System::Windows::Forms::GroupBox^  groupBox15;
	private: System::Windows::Forms::ComboBox^  comboBox_Kmeans_Option;

	private: System::Windows::Forms::Label^  label20;
	private: System::Windows::Forms::GroupBox^  groupBox16;
	private: System::Windows::Forms::CheckBox^  checkBox_ShowRange;
	private: System::Windows::Forms::GroupBox^  groupBox17;
	private: System::Windows::Forms::Label^  label22;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::Label^  label21;
	private: System::Windows::Forms::Label^  label19;
private: System::Windows::Forms::ComboBox^  comboBox_Weight;
private: System::Windows::Forms::GroupBox^  groupBox18;
private: System::Windows::Forms::ComboBox^  comboBox_P_Function;

private: System::Windows::Forms::GroupBox^  groupBox19;
private: System::Windows::Forms::ComboBox^  comboBox2;
private: System::Windows::Forms::TextBox^  textBox2;
private: System::Windows::Forms::Label^  label23;
private: System::Windows::Forms::GroupBox^  groupBox20;
private: System::Windows::Forms::GroupBox^  groupBox21;
private: System::Windows::Forms::Label^  label24;
private: System::Windows::Forms::TextBox^  textBox_initail;
private: System::Windows::Forms::ComboBox^  comboBox1;
private: System::Windows::Forms::Label^  label27;
private: System::Windows::Forms::Label^  label26;
private: System::Windows::Forms::TextBox^  textBox_bias;

private: System::Windows::Forms::Label^  label25;
private: System::Windows::Forms::TextBox^  textBox4;
private: System::Windows::Forms::TextBox^  textBox_Epsilon;

private: System::Windows::Forms::Label^  label28;


private: System::Windows::Forms::ComboBox^  comboBox_kNN;


		Brush^ ClassToColor(int c) {
			Brush^ gcBrush;
		switch (c)
			{
            case -2:
				gcBrush = gcnew SolidBrush(Color::Black);
                break;
            case -1:
 				gcBrush = gcnew SolidBrush(Color::Blue);
                break;
            case 0:
				gcBrush = gcnew SolidBrush(Color::Green);
                break;
            case 1:
				gcBrush = gcnew SolidBrush(Color::Red);
                break;
            case 2:
				gcBrush = gcnew SolidBrush(Color::Yellow);
                break;
            case 3:
				gcBrush = gcnew SolidBrush(Color::Orange);
                break;
            case 4:
				gcBrush = gcnew SolidBrush(Color::Brown);
                break;
            default:
				gcBrush = gcnew SolidBrush(Color::Purple);
                break;
				 }
		return gcBrush;
		}
		Pen^ ClassToPenColor(int c) {
				Pen^ gcPen;
		switch (c)
			{
				case -2:
						gcPen = gcnew Pen(Color::Black);
						break;
				case -1:
 						gcPen = gcnew Pen(Color::Blue);
						break;
				case 0:
						gcPen = gcnew Pen(Color::Green);
						break;
				case 1:
						gcPen = gcnew Pen(Color::Red);
						break;
				case 2:
 						gcPen = gcnew Pen(Color::Yellow);
						break;
				case 3:
						gcPen = gcnew Pen(Color::Orange);
						break;
				case 4:
						gcPen = gcnew Pen(Color::Brown);
						break;
				default:
						gcPen = gcnew Pen(Color::Purple);
				break;
				 }
			return gcPen;
			}

		void BayesMAP(){
			CalculateMeanSigma2();
			CalculateBayesianProb();
		}

		void NewPublicVariables(int MaxNumberOfData){
			InputData = new pData[MaxNumberOfData];
			PxyClass1 = new double[MaxNumberOfData];
			PxyClass2 = new double[MaxNumberOfData];

				//Clustering
			ClusterCenter= new pData[MaxNumClusters]; //ClusterCenter
			InputDataClusterType= new unsigned char[MaxNumberOfData];
			BackupClassKind = new int[MaxNumberOfData];
			Radius = new unsigned short[MaxNumClusters];
			dist= new double*[MaxNumberOfData]; //dist[][]-->distance between data and data
			for (int k=0; k<MaxNumberOfData; k++)
				dist[k]= new double[MaxNumberOfData];
			uij= new double*[MaxNumberOfData]; //dist[][]-->distance between data and data
			for (int k=0; k<MaxNumberOfData; k++)
				uij[k]= new double[MaxNumberOfData];
			ALLNNs= new short*[totalCTestData]; //dist[][]-->distance between data and data
			for (int k=0; k<totalCTestData; k++)
				ALLNNs[k]= new short[255];
			for(int  i= 0;i < totalCTestData;i++)
				for(int j = 0;j < 255;j++)
					ALLNNs[i][j] = 0;
			ALLCountClass1 = new short[totalCTestData];
			ALLCountClass2 = new short[totalCTestData];
			
			BDdist= new double*[totalRTestData]; //dist[][]-->distance between data and data
			for (int k=0; k<totalRTestData; k++)
				BDdist[k]= new double[255];
			NNs= new int*[totalRTestData]; //dist[][]-->distance between data and data
			for (int k=0; k<totalRTestData; k++)
				NNs[k]= new int[255];

			//Perceptron, LVQ
			W = new pData[MaxNumberOfData]; //Neural Networks(Perceptron, BP, LVQ)
		
		}

		void DeletePublicVariables(unsigned short MaxNumberOfData){
			delete [] InputData;
			delete [] PxyClass1;
			delete [] PxyClass2;

			//Clustering
			delete [] ClusterCenter;
			delete [] InputDataClusterType;
			delete [] BackupClassKind;
			delete [] Radius;
			for (int k=0; k<MaxNumberOfData; k++)
				delete [] dist[k];
			delete [] dist;
			for (int k=0; k<MaxNumberOfData; k++)
				delete [] uij[k];
			delete [] uij;
			for (int k=0; k<totalCTestData; k++)
				delete [] ALLNNs[k];
			delete [] ALLNNs;
			delete [] ALLCountClass1;
			delete [] ALLCountClass2;
			for (int k=0; k<totalRTestData; k++)
				delete [] BDdist[k];
			delete [] BDdist;
			for (int k=0; k<totalRTestData; k++)
				delete [] NNs[k];
			delete [] NNs;

			// Perceptron, LVQ
			delete[] W; //Neural Networks(Perceptron, BP, LVQ)
		}


		double Sgn(double Num1){
			return (Num1 >= 0.0) ? 1.0 : -1.0;
		}

		void CalculateMeanSigma2(){
			MeanX1 = 0.0; MeanY1 = 0.0;
			MeanX2 = 0.0; MeanY2 = 0.0;
			NumClass1 = 0.0; NumClass2 = 0.0;

			for(int i = 0;i < NumberOfData;i++){
				if(InputData[i].ClassKind == 1){
					MeanX1 += InputData[i].X;
					MeanY1 += InputData[i].Y;
					NumClass1++;
				}else{
					MeanX2 += InputData[i].X;
					MeanY2 += InputData[i].Y;
					NumClass2++;
				}
			}

			MeanX1 = MeanX1 / NumClass1; MeanY1 = MeanY1 / NumClass1;
			MeanX2 = MeanX2 / NumClass2; MeanY2 = MeanY2 / NumClass2;

			Sigma2X1 = 0.0; Sigma2Y1 = 0.0;
			Sigma2X2 = 0.0; Sigma2Y2 = 0.0;
			SigmaXY1 = 0.0; SigmaXY2 = 0.0;

			for(int i = 0;i < NumberOfData;i++){
				if(InputData[i].ClassKind == 1){
					Sigma2X1 += pow((InputData[i].X - MeanX1),2);
					Sigma2Y1 += pow((InputData[i].Y - MeanY1),2);
					SigmaXY1 += (InputData[i].X - MeanX1) * (InputData[i].Y - MeanY1);
				}else{
					Sigma2X2 += pow((InputData[i].X - MeanX2),2);
					Sigma2Y2 += pow((InputData[i].Y - MeanY2),2);
					SigmaXY2 += (InputData[i].X - MeanX2) * (InputData[i].Y - MeanY2);
				}
			}

			if(NumClass1 > 0){
				if(NumClass1 == 1 || checkBox_Unbiased->Checked){
					Sigma2X1 /= NumClass1;
					Sigma2Y1 /= NumClass1;
					SigmaXY1 /= NumClass1;
				}else{
					Sigma2X1 /= (NumClass1 - 1);
					Sigma2Y1 /= (NumClass1 - 1);
					SigmaXY1 /= (NumClass1 - 1);
				}
			}//if NumClass1 > 0
			if(NumClass2 == 1 || checkBox_Unbiased->Checked){
				Sigma2X2 /= NumClass2;
				Sigma2Y2 /= NumClass2;
				SigmaXY2 /= NumClass2;
			}else{
				Sigma2X2 /= (NumClass2 - 1);
				Sigma2Y2 /= (NumClass2 - 1);
				SigmaXY2 /= (NumClass2 - 1);
			}
		    
			SigmaX1 = sqrt(Sigma2X1); SigmaY1 = sqrt(Sigma2Y1);
			SigmaX2 = sqrt(Sigma2X2); SigmaY2 = sqrt(Sigma2Y2);
			Correlation1 = SigmaXY1 / (SigmaX1*SigmaY1); Correlation12 = Correlation1*Correlation1;
			Correlation2 = SigmaXY2 / (SigmaX2*SigmaY2); Correlation22 = Correlation2*Correlation2;
			detA1 = Sigma2X1*Sigma2Y1 - SigmaXY1 * SigmaXY1;
			detA2 = Sigma2X2*Sigma2Y2 - SigmaXY2 * SigmaXY2;
		}

		double evalPxy1(pData Data1){
			double dx, dy, dx2, dy2, c1, Ndist, tmp;
			//Class Red
			dx = Data1.X - MeanX1; dy = Data1.Y - MeanY1;
			dx2 = dx*dx; dy2 = dy*dy;

			if(Correlation12 != 1.0)
				c1 = 1.0 / ( 1.0 / Correlation12);
			else
				c1 = 1.0;

			if(detA1 == 0.0){
				Ndist = dx2 - 2.0*dx*dy + dy2;
				tmp = exp(-0.5*Ndist);
			}else{
				Ndist = dx2 / Sigma2X1 - 2.0*Correlation12*dx*dy / SigmaXY1 + dy2 / Sigma2Y1;
				tmp = exp(-0.5*c1*Ndist) /( 2.0*Pi*sqrt(detA1));
			}
			return tmp;
		}

		double evalPxy2(pData Data1){
			double dx, dy, dx2, dy2, c1, Ndist, tmp;
				//Class Red
			dx = Data1.X - MeanX2; dy = Data1.Y - MeanY2;
			dx2 = dx*dx; dy2 = dy*dy;

			if(Correlation22 != 1.0)
				c1 = 1.0 / ( 1.0 / Correlation22);
			else
				c1 = 1.0;

			if(detA2 == 0.0){
				Ndist = dx2 - 2.0*dx*dy + dy2;
				tmp = exp(-0.5*Ndist);
			}else{
				Ndist = dx2 / Sigma2X2 - 2.0*Correlation22*dx*dy / SigmaXY2 + dy2 / Sigma2Y2;
				tmp = exp(-0.5*c1*Ndist) /( 2.0*Pi*sqrt(detA2));
			}
			return tmp;
		}

		void CalculateBayesianProb(){
			double dx, dy, dx2, dy2, c1, Ndist;

			PClass1 = (double)NumClass1 / NumberOfData;
			PClass2 = 1.0 - PClass1;
			for(int i = 0;i < NumberOfData;i++){
				//Class1 Red
				dx = InputData[i].X - MeanX1; dy = InputData[i].Y - MeanY1;
				dx2 = dx*dx; dy2 = dy*dy;

				if(Correlation12 != 1.0)
					c1 = 1.0 / ( 1.0 / Correlation12);
				else
					c1 = 1.0;

				if(detA1 == 0.0){
					Ndist = dx2 - 2.0*dx*dy + dy2;
					PxyClass1[i] = exp(-0.5*Ndist);
				}else{
					Ndist = dx2 / Sigma2X1 - 2.0*Correlation12*dx*dy / SigmaXY1 + dy2 / Sigma2Y1;
					PxyClass1[i] = exp(-0.5*c1*Ndist) /( 2.0*Pi*sqrt(detA1));
				}
				//Class2 Blue
				dx = InputData[i].X - MeanX2; dy = InputData[i].Y - MeanY2;
				dx2 = dx*dx; dy2 = dy*dy;

				if(Correlation22 != 1.0)
					c1 = 1.0 / ( 1.0 / Correlation22);
				else
					c1 = 1.0;

				if(detA2 == 0.0){
					Ndist = dx2 - 2.0*dx*dy + dy2;
					PxyClass2[i] = exp(-0.5*Ndist);
				}else{
					Ndist = dx2 / Sigma2X2 - 2.0*Correlation22*dx*dy / SigmaXY2 + dy2 / Sigma2Y2;
					PxyClass2[i] = exp(-0.5*c1*Ndist) /( 2.0*Pi*sqrt(detA2));
				}
			}
		}

		void GaussEliminationPivot(int n){
			double tmp, pvt;
			int index_pvt, *pivot;

			pivot = new int[n];

			for (int j = 0; j < n-1; j++) {
				pvt = abs(A[j][j]);
				pivot[j] = j;
				index_pvt = j;
				//find pivot
				for (int i = j+1; i < n; i++) {
					if (abs(A[i][j]) > pvt) {
						pvt = abs(A[i][j]);
						index_pvt = i;
					}//if
				}//for i

				//switch row pivot[j] and row index_pvt
				if (pivot[j] != index_pvt) {
					for (int i = 0; i < n; i++) {
						tmp = A[pivot[j]][i];
						A[pivot[j]][i] = A[index_pvt][i];
						A[index_pvt][i] = tmp;
					}//for i
					tmp = B[pivot[j]];
					B[pivot[j]] = B[index_pvt];
					B[index_pvt] = tmp;
				}//if

				for (int i = j+1; i < n; i++)
					A[i][j] /= A[j][j];

				//produce Upper triangle matrix
				for (int i = j+1; i < n; i++) {
					for (int k = j+1; k < n; k++) {
						A[i][k] -= A[i][j]*A[j][k];
					}//for k
					B[i] -= A[i][j] * B[j];
				}//for i
			}//for j

			//back substitution
			//for (int i = 0; i < n; i++)
			//	NLcoef[i] =0.0;
			NLcoef[n-1] = B[n-1] / A[n-1][n-1];
			for (int j = n-2; j >= 0; j--) {
				NLcoef[j] = B[j];
				for (int k = n-1; k > j; k--) {
					NLcoef[j] -= NLcoef[k]*A[j][k];
				}//for k
				NLcoef[j] /= A[j][j];
			}//for j
			delete [] pivot;
		}  //¸ÑÁp¥ß¤èµ{¦¡¤§°ª´µ®ø¥hªk
		void LinearRegression(){
			double xy = 0.0;
			double xplus = 0.0;
			double yplus = 0.0;
			double xmult = 0.0;
			double ymult = 0.0;	
			for(int i = 0;i < NumberOfData;i++){
				xplus += InputData[i].X;
				yplus += InputData[i].Y;
				xy += InputData[i].X * InputData[i].Y;
				xmult += InputData[i].X * InputData[i].X;
				ymult += InputData[i].Y * InputData[i].Y;
			}
			LR_a1 = (NumberOfData*xy - xplus*yplus) / (NumberOfData*xmult - xplus*xplus);
			LR_a0 = (yplus / NumberOfData) - (LR_a1*xplus / NumberOfData);
					

		} //Linear --½u©Ê°jÂk¤½¥Îµ{¦¡
		void LinearRegressionLn(){
			double xy = 0.0;
			double xplus = 0.0;
			double yplus = 0.0;
			double xmult = 0.0;
			//double ymult = 0.0;	
			double lny = 0.0;
			double shiftX,shiftY;
			for(int i = 0;i < NumberOfData;i++){
				shiftX = InputData[i].X + 2.0;
				shiftY = InputData[i].Y + 2.0;
				xplus += shiftX;
				lny = log(shiftY);
				yplus += lny;
				xy += shiftX * lny;
				xmult += shiftX * shiftX;
				//ymult += InputData[i].Y * InputData[i].Y;
			}
			LR_a1 = (NumberOfData*xy - xplus*yplus) / (NumberOfData*xmult - xplus*xplus);
			LR_a0 = (yplus / NumberOfData) - (LR_a1*xplus / NumberOfData);
			LR_a0 = exp(LR_a0);
		} //Linear¡Xlog e ½u©Ê°jÂk§@«D½u©Ê°jÂk¤½¥Îµ{¦¡
		void NonlinearRegression(int degree){
			double xplus = 0.0;
			double xmult = 0.0;	
			double xpow3 = 0.0;
			double yplus = 0.0;
			double x0 = 0.0;
			

			for(int j = 0;j < degree+1;j++){
				for(int i =0;i < degree+1;i++){
					for(int k = 0;k < NumberOfData;k++){
						A[i][j] += pow(InputData[k].X,i+j);	
						
					}
				}
			}

			for(int j = 0;j < degree + 1;j++){
				for(int k = 0;k < NumberOfData;k++){
					B[j] += pow(InputData[k].X,j)*InputData[k].Y;
				}	
			}

			GaussEliminationPivot(degree+1);
		
		} //Non-linear¡X«D½u©Ê°jÂk¤½¥Îµ{¦¡
		double rand01() {
			//srand( (unsigned)time(NULL) );
			return rand() / double(RAND_MAX);
		} //²£¥Í¤¶©ó0.0~1.0¶¡ªº¶Ã¼Æ
		int rand_m(int Num1) {
			//srand((unsigned)time(NULL));
			return rand() % Num1;
		} //²£¥Í¤¶©ó0~Num1-1¶¡ªº¾ã¼Æ¶Ã¼Æ
		double MAX(double Num1, double Num2){
			if(Num1 > Num2)
				return Num1;
			return Num2;
		}//¨úNum1©MNum2¨âªÌ¸û¤jªº¼Æ
		double MIN(double Num1, double Num2){
			if(Num1 < Num2)
				return Num1;
			return Num2;
		}//¨úNum1©MNum2¨âªÌ¸û¤pªº¼Æ
		void K_Means(unsigned char K_Clusters){
			switch(comboBox_Kmeans_Option->SelectedIndex){
				case 0:
					int  classj;
					double  newDist;
					double min,datax = 0,datay = 0;
					double delta = Convert::ToDouble(textBox_delta->Text);
					double newDelta = delta;
					

					for(int i = 0;i < K_Clusters;i++){
						int ran = rand_m(NumberOfData);
						ClusterCenter[i] = InputData[ran];
						ClusterCenter[i].ClassKind = i;
					}

					while(newDelta >= delta ){
						newDelta = 0.0;
						for(int i = 0;i < NumberOfData;i++){
							for(int j = 0;j < K_Clusters;j++)
								dist[i][j] = pow(InputData[i].X - ClusterCenter[j].X,2)+pow(InputData[i].Y - ClusterCenter[j].Y,2);
						}	
						
						for(int i = 0;i < NumberOfData;i++){
							min = dist[i][0];
							for(int j = 0;j < K_Clusters;j++){
								min = MIN(min,dist[i][j]);
							}


							for(int j =0;j < K_Clusters;j++){
								if(min == dist[i][j]){
									InputData[i].ClassKind = j;
									dist[i][0] = dist[i][j];
									
									break;									
								}
							}
						}

						for(int j = 0;j < K_Clusters;j++){
							datax = 0;
							datay = 0;
							classj = 0;
							
							for(int i = 0;i < NumberOfData;i++){
								if(InputData[i].ClassKind == j){
									classj++;
									datax += InputData[i].X;
									datay += InputData[i].Y;				
								}	
							}
							datax = datax/classj;
							datay = datay/classj;
							newDist = sqrt(pow(datax-ClusterCenter[j].X,2) + pow(datay-ClusterCenter[j].Y,2));
							ClusterCenter[j].X = datax;
							ClusterCenter[j].Y = datay;
							
							newDelta = MAX(newDelta,newDist);
						}
					}
				
					break;
			}
					
			
			
		} //k_means¤½¥Îµ{¦¡
		void FCM(unsigned char K_Clusters){
			switch(comboBox_Kmeans_Option->SelectedIndex){
				case 0:
					int b = 2;
					double a,uS,max,datax = 0,datay = 0;
					double newDist = 0, distS ;
					double delta = Convert::ToDouble(textBox_delta->Text);
					double newDelta = delta;
					a = 1.0/(double)(b-1);

					

					for(int i = 0;i < K_Clusters;i++){
						int ran = rand_m(NumberOfData);
						ClusterCenter[i] = InputData[ran];
						ClusterCenter[i].ClassKind = i;
					}

					while(newDelta >= delta ){
						newDelta = 0.0;
						
						for(int i = 0;i < NumberOfData;i++){
							distS = 0.0;
							for(int j = 0;j < K_Clusters;j++){
								dist[i][j] = sqrt(pow(InputData[i].X - ClusterCenter[j].X,2)+pow(InputData[i].Y - ClusterCenter[j].Y,2));	
								if(dist[i][j] == 0.0)
									dist[i][j] = 1.0e38;
								uij[i][j] = pow(1.0/dist[i][j],a);
								distS += uij[i][j];
							}
							

							for(int j = 0;j < K_Clusters;j++){
								uij[i][j] = uij[i][j]/distS;
								
							}

						}
						
						for(int j = 0;j < K_Clusters;j++){
							double xt = 0;
							double yt = 0;
							uS = 0.0;
							for(int i = 0;i < NumberOfData;i++){
								xt += pow(uij[i][j],b)*InputData[i].X;
								yt += pow(uij[i][j],b)*InputData[i].Y;
								uS += pow(uij[i][j],b);
							}
							datax = xt / uS;
							datay = yt / uS;
							newDist = sqrt(pow(datax-ClusterCenter[j].X,2) + pow(datay-ClusterCenter[j].Y,2));
							ClusterCenter[j].X = datax;
							ClusterCenter[j].Y = datay;
							
							newDelta = MAX(newDelta,newDist);
						}
					}
					for(int i = 0;i < NumberOfData;i++){
							max = uij[i][0];
							for(int j = 0;j < K_Clusters;j++){
								max = MAX(max,uij[i][j]);
							}


							for(int j =0;j < K_Clusters;j++){
								if(max == uij[i][j]){
									InputData[i].ClassKind = j;
									dist[i][0] = uij[i][j];
									break;									
								}
							}
						}

					for(int i = 0;i < NumberOfData;i++){
						for(int j = 0;j < K_Clusters;j++){
							dist[i][j] = pow(InputData[i].X - ClusterCenter[j].X,2)+pow(InputData[i].Y - ClusterCenter[j].Y,2);	
						}
					}

				break;	
			}
		}
		short FindMaxKNN(){
			if(NumberOfData % 2 == 0)
				MaxKNN = NumberOfData-2;
			else
				MaxKNN = NumberOfData-1;
			return MaxKNN;
		} //­pºâ¨C­Ó¸ê®ÆÀÉk-NNªº¤W­­¡A§Yk³Ì¤j­È·|¨Ì¸ê®Æ¼Æ¶q¦Ó²§¡C
		void Create_kNN_Contour_Table() {
			
			double  mindis,*tmpd,shiftx,shifty;
			int *tmpi;
			bool change;
			tmpi = new int[kNNs];
			tmpd = new double[totalCTestData];

			for(int y = 0;y < imH ; y++){
				for(int  x= 0;x < imW ; x++){
					int ti = y*imW+x;
						ALLCountClass1[ti] = 0;
						ALLCountClass2[ti] = 0;
				}
			}

			for(int y = 0;y < imH ; y++){
				for(int  x= 0;x < imW ; x++){
					shiftx = (double)(x-CenterX)/CenterX;
					shifty = (double)(CenterY-y)/CenterY;
					int ti = y*imW+x;

					for(int i = 0;i < NumberOfData;i++)
						tmpd[i] = pow(InputData[i].X -shiftx,2) + pow(InputData[i].Y - shifty,2);

					for(int k = 0;k < kNNs;k++){
						mindis = tmpd[0];
						change = false;
						
						for(int i = 0;i < NumberOfData;i++){
							if(MIN(tmpd[i],mindis) == tmpd[i]){
								ALLNNs[ti][k] = i;
								mindis = tmpd[i];
								change = true;
							}
						}
						if(!change){
							ALLNNs[ti][k] = 0;
							tmpi = 0;
						}
						tmpd[ALLNNs[ti][k]] = 1e38;

						if(InputData[ALLNNs[ti][k]].ClassKind == 1)	
							ALLCountClass1[ti]++;
						else
							ALLCountClass2[ti]++;
					}	
				}
			}
			delete [] tmpd;
			delete [] tmpi;
			
		} //showContour®É©Ò»Ý­««Øªº¨C­Ó¸ê®ÆÂI¤§k-NN Table¡C
		void BuildAllkNNRegTable() {
			double  mindis,*tmpd,shiftx,shifty;
			int tmpi;
			bool change;
			//tmpi = new int[kNNs];
			//tmpd = new double[totalCTestData];

				for(int  x= 0;x < imW ; x++){
					shiftx = (double)(x-CenterX)/CenterX;
					
					for(int i = 0;i < NumberOfData;i++)
						BDdist[x][i] = abs(InputData[i].X -shiftx);

					for(int k = 0;k < kNNs;k++){
						mindis = BDdist[x][0];
						change = false;
						
						for(int i = 0;i < NumberOfData;i++){
							if(MIN(BDdist[x][i],mindis) == BDdist[x][i]){
								NNs[x][k] = i;
								mindis = BDdist[x][i];
								change = true;
							}
						}
						if(!change){
							NNs[x][k] = 0;
							//tmpi = 0;
						}
						BDdist[x][NNs[x][k]] = 1e38;
					}	
				}
			
			//delete [] tmpd;
			//delete [] tmpi;
		} //showRegression®É©Ò»Ý­««Øªº¨C­Ó¸ê®ÆÂI¤§k-NN Table¡C
		double PerceptronClassify(pData Sample){
			double n =( Sample.X * W[0].X + Sample.Y * W[0].Y )+ Bias;
			return Sgn(n);
		} //­pºâ¨C­Ó´ú¸Õ¸ê®ÆÄÝ©ó­þ¤@Ãþ?
		void Perceptron_Ctrain(){
			W[0].X = rand01()*2 -1;
			W[0].Y = rand01()*2-1;
			Bias = Convert::ToDouble(textBox_bias->Text);
			double a;
			int c =0;
			double alpha = Convert::ToDouble(textBox_initail->Text);
			int e;
			int Se = 1;
			while(Se != 0 || c < Convert::ToInt32(textBox_MaxIter->Text)){
				Se = 0;
				for(int i = 0;i <NumberOfData;i++){
					a = PerceptronClassify(InputData[i]);
					e = InputData[i].ClassKind - a;
					W[0].X += e*InputData[i].X*alpha;
					W[0].Y +=  e*InputData[i].Y*alpha;
					Bias +=  e*alpha;
					Se += abs(e);
				}
				c++;	
			}
			
		} //Training weights for Perceptron Classification¡C
		double PerceptronRegression(double Sample, int T_Function){
			double ty = (Sample * W[0].X )+ Bias; 
			double  n2 = -2*(Sample*W[0].X +Bias);
			switch(T_Function){
			case 1:
				return ty;
				break;
			case 3:
				ty = 2.0/(1+exp(n2))-1;
				return ty;
			}
			

		} //­pºâ¨C­Ó´ú¸Õ¸ê®Æ°jÂk­È?
		void Perceptron_RTrain(int T_Function){
			double alpha = Convert::ToDouble(textBox_initail->Text);
			Bias = Convert::ToDouble(textBox_bias->Text);
			double targety;
			int c = 0;
			W[0].X = rand01()*2 -1;
			double E = 1e20,erro;
			switch(T_Function){
			case 1:
				while(E > Convert::ToDouble(textBox_Epsilon->Text) && c < Convert::ToInt32(textBox_MaxIter->Text)){
					E = 0.0;
					for(int i = 0;i < NumberOfData;i++){
						targety = PerceptronRegression(InputData[i].X,T_Function);
						erro = InputData[i].Y - targety;
						E += 0.5 * pow(erro,2);
						W[0].X += erro * InputData[i].X*alpha;
						Bias += erro*alpha;
					}
					c++;
				}
				break;
			case 3:
				while(E > Convert::ToDouble(textBox_Epsilon->Text) && c < Convert::ToInt32(textBox_MaxIter->Text)){
					E = 0.0;
					for(int i = 0;i < NumberOfData;i++){
						targety = PerceptronRegression(InputData[i].X,T_Function);
						erro = InputData[i].Y - targety;
						E += 0.5 * pow(erro,2);
						erro = (InputData[i].Y - targety)*(1-targety*targety);
						W[0].X += erro * InputData[i].X;
						Bias += erro*alpha;
					}
					c++;
				}
				break;
			}
		} // Training weights for Perceptron Regression¡C
	
	public: 
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: ¦b¦¹¥[¤J«Øºc¨ç¦¡µ{¦¡½X
			//
		}

	protected:
		/// <summary>
		/// ²M°£¥ô¦ó¨Ï¥Î¤¤ªº¸ê·½¡C
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	protected: 
	private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  imageToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  dataEditToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  applicationsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  aboutToolStripMenuItem;
	private: System::Windows::Forms::ToolStrip^  toolStrip1;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton1;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton2;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton3;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton4;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton5;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton6;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton7;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton8;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton9;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton10;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton11;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton12;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton13;
	private: System::Windows::Forms::CheckBox^  checkBox1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TextBox^  textBox_X;

	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::TextBox^  textBox_Y;

	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::ComboBox^  comboBox_psize;

	private: System::Windows::Forms::GroupBox^  groupBox2;
	private: System::Windows::Forms::TextBox^  textBox_datasize;

	private: System::Windows::Forms::GroupBox^  groupBox3;
	private: System::Windows::Forms::TextBox^  textBox_MaxSize;
	private: System::Windows::Forms::Button^  button_clear;
private: System::Windows::Forms::Button^  Run;



	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::GroupBox^  groupBox4;
	private: System::Windows::Forms::RadioButton^  radioButton2;
	private: System::Windows::Forms::RadioButton^  radioButton_Single;


	private: System::Windows::Forms::GroupBox^  groupBox5;
	private: System::Windows::Forms::ComboBox^  comboBox_CS;
	private: System::Windows::Forms::RadioButton^  radioButton_NC;


	private: System::Windows::Forms::RadioButton^  radioButton_C2;
private: System::Windows::Forms::RadioButton^  radioButton_CS;



	private: System::Windows::Forms::RadioButton^  radioButton_C1;

	private: System::Windows::Forms::GroupBox^  groupBox6;
	private: System::Windows::Forms::HScrollBar^  hScrollBar1;
	private: System::Windows::Forms::TextBox^  textBox8;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::TextBox^  textBox7;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::TextBox^  textBox6;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::TextBox^  textBox5;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::GroupBox^  groupBox7;
	private: System::Windows::Forms::RadioButton^  radioButton10;
	private: System::Windows::Forms::RadioButton^  radioButton8;
	private: System::Windows::Forms::Label^  label7;
private: System::Windows::Forms::TextBox^  textBox_Filename;

	private: System::Windows::Forms::RichTextBox^  richTextBox1;
	private: System::Windows::Forms::FontDialog^  fontDialog1;
	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::Label^  label9;
	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::Label^  label11;
	private: System::Windows::Forms::Label^  label12;
	private: System::Windows::Forms::Label^  label13;
	private: System::ComponentModel::IContainer^  components;

	private: System::Windows::Forms::SaveFileDialog^  saveFileDialog1;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;

	private:
		/// <summary>
		/// ³]­p¤u¨ã©Ò»ÝªºÅÜ¼Æ¡C
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// ¦¹¬°³]­p¤u¨ã¤ä´©©Ò»Ýªº¤èªk - ½Ð¤Å¨Ï¥Îµ{¦¡½X½s¿è¾¹
		/// ­×§ï³o­Ó¤èªkªº¤º®e¡C
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(Form1::typeid));
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveAsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->newFileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->imageToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->clearImageToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showResultToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showContourToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showMeansToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showRegressionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showClusteredToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showClusterCenterToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->dataEditToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->applicationsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->aboutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStrip1 = (gcnew System::Windows::Forms::ToolStrip());
			this->toolStripButton1 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton2 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton3 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton4 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton5 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton6 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton7 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton8 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton9 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton10 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton11 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton12 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton13 = (gcnew System::Windows::Forms::ToolStripButton());
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->textBox_X = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->textBox_Y = (gcnew System::Windows::Forms::TextBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_psize = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_datasize = (gcnew System::Windows::Forms::TextBox());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_MaxSize = (gcnew System::Windows::Forms::TextBox());
			this->button_clear = (gcnew System::Windows::Forms::Button());
			this->Run = (gcnew System::Windows::Forms::Button());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButton2 = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_Single = (gcnew System::Windows::Forms::RadioButton());
			this->groupBox5 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_CS = (gcnew System::Windows::Forms::ComboBox());
			this->radioButton_NC = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_C2 = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_CS = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_C1 = (gcnew System::Windows::Forms::RadioButton());
			this->groupBox6 = (gcnew System::Windows::Forms::GroupBox());
			this->hScrollBar1 = (gcnew System::Windows::Forms::HScrollBar());
			this->textBox8 = (gcnew System::Windows::Forms::TextBox());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->textBox7 = (gcnew System::Windows::Forms::TextBox());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->textBox6 = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->textBox5 = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->groupBox7 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButton10 = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton8 = (gcnew System::Windows::Forms::RadioButton());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->textBox_Filename = (gcnew System::Windows::Forms::TextBox());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->fontDialog1 = (gcnew System::Windows::Forms::FontDialog());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->groupBox8 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_Run = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox9 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_classify = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox10 = (gcnew System::Windows::Forms::GroupBox());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->comboBox_clusters = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox_clustering = (gcnew System::Windows::Forms::ComboBox());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->groupBox11 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_NL_degree = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox_regression = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox12 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox13 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_MaxIter = (gcnew System::Windows::Forms::TextBox());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->textBox_delta = (gcnew System::Windows::Forms::TextBox());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->groupBox14 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->checkBox_Unbiased = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox15 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_Kmeans_Option = (gcnew System::Windows::Forms::ComboBox());
			this->label20 = (gcnew System::Windows::Forms::Label());
			this->groupBox16 = (gcnew System::Windows::Forms::GroupBox());
			this->checkBox_ShowRange = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox17 = (gcnew System::Windows::Forms::GroupBox());
			this->label22 = (gcnew System::Windows::Forms::Label());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label21 = (gcnew System::Windows::Forms::Label());
			this->label19 = (gcnew System::Windows::Forms::Label());
			this->comboBox_Weight = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox_kNN = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox18 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_P_Function = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox19 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox2 = (gcnew System::Windows::Forms::ComboBox());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->label23 = (gcnew System::Windows::Forms::Label());
			this->groupBox20 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_Epsilon = (gcnew System::Windows::Forms::TextBox());
			this->label28 = (gcnew System::Windows::Forms::Label());
			this->groupBox21 = (gcnew System::Windows::Forms::GroupBox());
			this->label27 = (gcnew System::Windows::Forms::Label());
			this->label26 = (gcnew System::Windows::Forms::Label());
			this->textBox_bias = (gcnew System::Windows::Forms::TextBox());
			this->label25 = (gcnew System::Windows::Forms::Label());
			this->textBox4 = (gcnew System::Windows::Forms::TextBox());
			this->label24 = (gcnew System::Windows::Forms::Label());
			this->textBox_initail = (gcnew System::Windows::Forms::TextBox());
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->menuStrip1->SuspendLayout();
			this->toolStrip1->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->pictureBox1))->BeginInit();
			this->groupBox4->SuspendLayout();
			this->groupBox5->SuspendLayout();
			this->groupBox6->SuspendLayout();
			this->groupBox7->SuspendLayout();
			this->groupBox8->SuspendLayout();
			this->groupBox9->SuspendLayout();
			this->groupBox10->SuspendLayout();
			this->groupBox11->SuspendLayout();
			this->groupBox12->SuspendLayout();
			this->groupBox13->SuspendLayout();
			this->groupBox14->SuspendLayout();
			this->groupBox15->SuspendLayout();
			this->groupBox16->SuspendLayout();
			this->groupBox17->SuspendLayout();
			this->groupBox18->SuspendLayout();
			this->groupBox19->SuspendLayout();
			this->groupBox20->SuspendLayout();
			this->groupBox21->SuspendLayout();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {this->fileToolStripMenuItem, 
				this->imageToolStripMenuItem, this->dataEditToolStripMenuItem, this->applicationsToolStripMenuItem, this->aboutToolStripMenuItem});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Padding = System::Windows::Forms::Padding(7, 2, 0, 2);
			this->menuStrip1->Size = System::Drawing::Size(1194, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {this->openToolStripMenuItem, 
				this->saveToolStripMenuItem, this->saveAsToolStripMenuItem, this->exitToolStripMenuItem, this->newFileToolStripMenuItem});
			this->fileToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// openToolStripMenuItem
			// 
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(119, 22);
			this->openToolStripMenuItem->Text = L"Open";
			this->openToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::openToolStripMenuItem_Click);
			// 
			// saveToolStripMenuItem
			// 
			this->saveToolStripMenuItem->Name = L"saveToolStripMenuItem";
			this->saveToolStripMenuItem->Size = System::Drawing::Size(119, 22);
			this->saveToolStripMenuItem->Text = L"Save";
			this->saveToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveToolStripMenuItem_Click);
			// 
			// saveAsToolStripMenuItem
			// 
			this->saveAsToolStripMenuItem->Name = L"saveAsToolStripMenuItem";
			this->saveAsToolStripMenuItem->Size = System::Drawing::Size(119, 22);
			this->saveAsToolStripMenuItem->Text = L"Save as";
			this->saveAsToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveAsToolStripMenuItem_Click);
			// 
			// exitToolStripMenuItem
			// 
			this->exitToolStripMenuItem->Name = L"exitToolStripMenuItem";
			this->exitToolStripMenuItem->Size = System::Drawing::Size(119, 22);
			this->exitToolStripMenuItem->Text = L"Exit";
			this->exitToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::exitToolStripMenuItem_Click);
			// 
			// newFileToolStripMenuItem
			// 
			this->newFileToolStripMenuItem->Name = L"newFileToolStripMenuItem";
			this->newFileToolStripMenuItem->Size = System::Drawing::Size(119, 22);
			this->newFileToolStripMenuItem->Text = L"New File";
			// 
			// imageToolStripMenuItem
			// 
			this->imageToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(7) {this->clearImageToolStripMenuItem, 
				this->showResultToolStripMenuItem, this->showContourToolStripMenuItem, this->showMeansToolStripMenuItem, this->showRegressionToolStripMenuItem, 
				this->showClusteredToolStripMenuItem, this->showClusterCenterToolStripMenuItem});
			this->imageToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->imageToolStripMenuItem->Name = L"imageToolStripMenuItem";
			this->imageToolStripMenuItem->Size = System::Drawing::Size(51, 20);
			this->imageToolStripMenuItem->Text = L"Image";
			this->imageToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::imageToolStripMenuItem_Click);
			// 
			// clearImageToolStripMenuItem
			// 
			this->clearImageToolStripMenuItem->Name = L"clearImageToolStripMenuItem";
			this->clearImageToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->clearImageToolStripMenuItem->Text = L"Clear Image";
			this->clearImageToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::clearImageToolStripMenuItem_Click);
			// 
			// showResultToolStripMenuItem
			// 
			this->showResultToolStripMenuItem->Name = L"showResultToolStripMenuItem";
			this->showResultToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showResultToolStripMenuItem->Text = L"Show Result";
			this->showResultToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showResultToolStripMenuItem_Click);
			// 
			// showContourToolStripMenuItem
			// 
			this->showContourToolStripMenuItem->Name = L"showContourToolStripMenuItem";
			this->showContourToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showContourToolStripMenuItem->Text = L"Show Contour";
			this->showContourToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showContourToolStripMenuItem_Click);
			// 
			// showMeansToolStripMenuItem
			// 
			this->showMeansToolStripMenuItem->Name = L"showMeansToolStripMenuItem";
			this->showMeansToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showMeansToolStripMenuItem->Text = L"Show Means";
			this->showMeansToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showMeansToolStripMenuItem_Click);
			// 
			// showRegressionToolStripMenuItem
			// 
			this->showRegressionToolStripMenuItem->Name = L"showRegressionToolStripMenuItem";
			this->showRegressionToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showRegressionToolStripMenuItem->Text = L"Show Regression";
			this->showRegressionToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showRegressionToolStripMenuItem_Click);
			// 
			// showClusteredToolStripMenuItem
			// 
			this->showClusteredToolStripMenuItem->Name = L"showClusteredToolStripMenuItem";
			this->showClusteredToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showClusteredToolStripMenuItem->Text = L"Show Clustered";
			this->showClusteredToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showClusteredToolStripMenuItem_Click);
			// 
			// showClusterCenterToolStripMenuItem
			// 
			this->showClusterCenterToolStripMenuItem->Name = L"showClusterCenterToolStripMenuItem";
			this->showClusterCenterToolStripMenuItem->Size = System::Drawing::Size(179, 22);
			this->showClusterCenterToolStripMenuItem->Text = L"Show Cluster Center";
			this->showClusterCenterToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showClusterCenterToolStripMenuItem_Click);
			// 
			// dataEditToolStripMenuItem
			// 
			this->dataEditToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->showDataToolStripMenuItem});
			this->dataEditToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->dataEditToolStripMenuItem->Name = L"dataEditToolStripMenuItem";
			this->dataEditToolStripMenuItem->Size = System::Drawing::Size(63, 20);
			this->dataEditToolStripMenuItem->Text = L"DataEdit";
			this->dataEditToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::dataEditToolStripMenuItem_Click);
			// 
			// showDataToolStripMenuItem
			// 
			this->showDataToolStripMenuItem->Name = L"showDataToolStripMenuItem";
			this->showDataToolStripMenuItem->Size = System::Drawing::Size(130, 22);
			this->showDataToolStripMenuItem->Text = L"Show Data";
			this->showDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showDataToolStripMenuItem_Click);
			// 
			// applicationsToolStripMenuItem
			// 
			this->applicationsToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, 
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->applicationsToolStripMenuItem->Name = L"applicationsToolStripMenuItem";
			this->applicationsToolStripMenuItem->Size = System::Drawing::Size(85, 20);
			this->applicationsToolStripMenuItem->Text = L"Applications";
			// 
			// aboutToolStripMenuItem
			// 
			this->aboutToolStripMenuItem->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->aboutToolStripMenuItem->Name = L"aboutToolStripMenuItem";
			this->aboutToolStripMenuItem->Size = System::Drawing::Size(52, 20);
			this->aboutToolStripMenuItem->Text = L"About";
			// 
			// toolStrip1
			// 
			this->toolStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(13) {this->toolStripButton1, 
				this->toolStripButton2, this->toolStripButton3, this->toolStripButton4, this->toolStripButton5, this->toolStripButton6, this->toolStripButton7, 
				this->toolStripButton8, this->toolStripButton9, this->toolStripButton10, this->toolStripButton11, this->toolStripButton12, this->toolStripButton13});
			this->toolStrip1->Location = System::Drawing::Point(0, 24);
			this->toolStrip1->Name = L"toolStrip1";
			this->toolStrip1->Size = System::Drawing::Size(1194, 25);
			this->toolStrip1->TabIndex = 1;
			this->toolStrip1->Text = L"toolStrip1";
			// 
			// toolStripButton1
			// 
			this->toolStripButton1->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton1->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton1->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton1.Image")));
			this->toolStripButton1->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton1->Name = L"toolStripButton1";
			this->toolStripButton1->Size = System::Drawing::Size(23, 22);
			this->toolStripButton1->Text = L"New";
			this->toolStripButton1->Click += gcnew System::EventHandler(this, &Form1::toolStripButton1_Click);
			// 
			// toolStripButton2
			// 
			this->toolStripButton2->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton2->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton2->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton2.Image")));
			this->toolStripButton2->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton2->Name = L"toolStripButton2";
			this->toolStripButton2->Size = System::Drawing::Size(23, 22);
			this->toolStripButton2->Text = L"Save";
			this->toolStripButton2->Click += gcnew System::EventHandler(this, &Form1::toolStripButton2_Click);
			// 
			// toolStripButton3
			// 
			this->toolStripButton3->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton3->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton3->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton3.Image")));
			this->toolStripButton3->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton3->Name = L"toolStripButton3";
			this->toolStripButton3->Size = System::Drawing::Size(23, 22);
			this->toolStripButton3->Text = L"Exit";
			this->toolStripButton3->Click += gcnew System::EventHandler(this, &Form1::toolStripButton3_Click);
			// 
			// toolStripButton4
			// 
			this->toolStripButton4->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton4->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton4->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton4.Image")));
			this->toolStripButton4->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton4->Name = L"toolStripButton4";
			this->toolStripButton4->Size = System::Drawing::Size(23, 22);
			this->toolStripButton4->Text = L"Open";
			this->toolStripButton4->Click += gcnew System::EventHandler(this, &Form1::toolStripButton4_Click);
			// 
			// toolStripButton5
			// 
			this->toolStripButton5->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton5->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton5->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton5.Image")));
			this->toolStripButton5->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton5->Name = L"toolStripButton5";
			this->toolStripButton5->Size = System::Drawing::Size(23, 22);
			this->toolStripButton5->Text = L"toolStripButton5";
			// 
			// toolStripButton6
			// 
			this->toolStripButton6->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton6->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton6->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton6.Image")));
			this->toolStripButton6->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton6->Name = L"toolStripButton6";
			this->toolStripButton6->Size = System::Drawing::Size(23, 22);
			this->toolStripButton6->Text = L"Save as";
			this->toolStripButton6->Click += gcnew System::EventHandler(this, &Form1::toolStripButton6_Click);
			// 
			// toolStripButton7
			// 
			this->toolStripButton7->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton7->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton7->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton7.Image")));
			this->toolStripButton7->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton7->Name = L"toolStripButton7";
			this->toolStripButton7->Size = System::Drawing::Size(23, 22);
			this->toolStripButton7->Text = L"Undo";
			// 
			// toolStripButton8
			// 
			this->toolStripButton8->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton8->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton8->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton8.Image")));
			this->toolStripButton8->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton8->Name = L"toolStripButton8";
			this->toolStripButton8->Size = System::Drawing::Size(23, 22);
			this->toolStripButton8->Text = L"Redo";
			// 
			// toolStripButton9
			// 
			this->toolStripButton9->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton9->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton9->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton9.Image")));
			this->toolStripButton9->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton9->Name = L"toolStripButton9";
			this->toolStripButton9->Size = System::Drawing::Size(23, 22);
			this->toolStripButton9->Text = L"Cut";
			// 
			// toolStripButton10
			// 
			this->toolStripButton10->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton10->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton10->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton10.Image")));
			this->toolStripButton10->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton10->Name = L"toolStripButton10";
			this->toolStripButton10->Size = System::Drawing::Size(23, 22);
			this->toolStripButton10->Text = L"Copy";
			// 
			// toolStripButton11
			// 
			this->toolStripButton11->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton11->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton11->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton11.Image")));
			this->toolStripButton11->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton11->Name = L"toolStripButton11";
			this->toolStripButton11->Size = System::Drawing::Size(23, 22);
			this->toolStripButton11->Text = L"Paste";
			// 
			// toolStripButton12
			// 
			this->toolStripButton12->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton12->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton12->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton12.Image")));
			this->toolStripButton12->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton12->Name = L"toolStripButton12";
			this->toolStripButton12->Size = System::Drawing::Size(23, 22);
			this->toolStripButton12->Text = L"toolStripButton12";
			// 
			// toolStripButton13
			// 
			this->toolStripButton13->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton13->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->toolStripButton13->Image = (cli::safe_cast<System::Drawing::Image^  >(resources->GetObject(L"toolStripButton13.Image")));
			this->toolStripButton13->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton13->Name = L"toolStripButton13";
			this->toolStripButton13->Size = System::Drawing::Size(23, 22);
			this->toolStripButton13->Text = L"toolStripButton13";
			// 
			// checkBox1
			// 
			this->checkBox1->AutoSize = true;
			this->checkBox1->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->checkBox1->Location = System::Drawing::Point(15, 75);
			this->checkBox1->Margin = System::Windows::Forms::Padding(3, 5, 3, 5);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(89, 21);
			this->checkBox1->TabIndex = 2;
			this->checkBox1->Text = L"Show Data";
			this->checkBox1->UseVisualStyleBackColor = true;
			this->checkBox1->CheckedChanged += gcnew System::EventHandler(this, &Form1::checkBox1_CheckedChanged);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(13, 108);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(19, 17);
			this->label1->TabIndex = 3;
			this->label1->Text = L"X:";
			// 
			// textBox_X
			// 
			this->textBox_X->Location = System::Drawing::Point(36, 104);
			this->textBox_X->Name = L"textBox_X";
			this->textBox_X->Size = System::Drawing::Size(86, 25);
			this->textBox_X->TabIndex = 4;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(128, 108);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(18, 17);
			this->label2->TabIndex = 5;
			this->label2->Text = L"Y:";
			// 
			// textBox_Y
			// 
			this->textBox_Y->Location = System::Drawing::Point(152, 104);
			this->textBox_Y->Name = L"textBox_Y";
			this->textBox_Y->Size = System::Drawing::Size(100, 25);
			this->textBox_Y->TabIndex = 6;
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->comboBox_psize);
			this->groupBox1->Location = System::Drawing::Point(258, 75);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(78, 56);
			this->groupBox1->TabIndex = 7;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"PointSize";
			// 
			// comboBox_psize
			// 
			this->comboBox_psize->FormattingEnabled = true;
			this->comboBox_psize->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"1", L"2", L"3"});
			this->comboBox_psize->Location = System::Drawing::Point(17, 25);
			this->comboBox_psize->Name = L"comboBox_psize";
			this->comboBox_psize->Size = System::Drawing::Size(47, 25);
			this->comboBox_psize->TabIndex = 0;
			this->comboBox_psize->Text = L"1";
			this->comboBox_psize->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_psize_SelectedIndexChanged);
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->textBox_datasize);
			this->groupBox2->Location = System::Drawing::Point(342, 75);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(80, 56);
			this->groupBox2->TabIndex = 8;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"DataSize";
			// 
			// textBox_datasize
			// 
			this->textBox_datasize->Location = System::Drawing::Point(10, 23);
			this->textBox_datasize->Name = L"textBox_datasize";
			this->textBox_datasize->Size = System::Drawing::Size(58, 25);
			this->textBox_datasize->TabIndex = 0;
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->textBox_MaxSize);
			this->groupBox3->Location = System::Drawing::Point(428, 75);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(80, 56);
			this->groupBox3->TabIndex = 9;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"MaxSize";
			// 
			// textBox_MaxSize
			// 
			this->textBox_MaxSize->Location = System::Drawing::Point(10, 23);
			this->textBox_MaxSize->Name = L"textBox_MaxSize";
			this->textBox_MaxSize->Size = System::Drawing::Size(58, 25);
			this->textBox_MaxSize->TabIndex = 0;
			this->textBox_MaxSize->Text = L"3000";
			this->textBox_MaxSize->TextChanged += gcnew System::EventHandler(this, &Form1::textBox_MaxSize_TextChanged);
			// 
			// button_clear
			// 
			this->button_clear->Location = System::Drawing::Point(523, 75);
			this->button_clear->Name = L"button_clear";
			this->button_clear->Size = System::Drawing::Size(50, 57);
			this->button_clear->TabIndex = 10;
			this->button_clear->Text = L"Clear";
			this->button_clear->UseVisualStyleBackColor = true;
			this->button_clear->Click += gcnew System::EventHandler(this, &Form1::button_Clear_Click);
			// 
			// Run
			// 
			this->Run->Location = System::Drawing::Point(579, 75);
			this->Run->Name = L"Run";
			this->Run->Size = System::Drawing::Size(52, 57);
			this->Run->TabIndex = 11;
			this->Run->Text = L"Run";
			this->Run->UseVisualStyleBackColor = true;
			this->Run->Click += gcnew System::EventHandler(this, &Form1::run_Click);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(36, 137);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(512, 512);
			this->pictureBox1->TabIndex = 12;
			this->pictureBox1->TabStop = false;
			this->pictureBox1->Click += gcnew System::EventHandler(this, &Form1::pictureBox1_Click);
			this->pictureBox1->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseClick);
			this->pictureBox1->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseDown);
			this->pictureBox1->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseMove);
			// 
			// groupBox4
			// 
			this->groupBox4->Controls->Add(this->radioButton2);
			this->groupBox4->Controls->Add(this->radioButton_Single);
			this->groupBox4->Location = System::Drawing::Point(554, 138);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Size = System::Drawing::Size(135, 56);
			this->groupBox4->TabIndex = 13;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"Input Model";
			// 
			// radioButton2
			// 
			this->radioButton2->AutoSize = true;
			this->radioButton2->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton2->Location = System::Drawing::Point(69, 25);
			this->radioButton2->Name = L"radioButton2";
			this->radioButton2->Size = System::Drawing::Size(63, 21);
			this->radioButton2->TabIndex = 1;
			this->radioButton2->TabStop = true;
			this->radioButton2->Text = L"Group";
			this->radioButton2->UseVisualStyleBackColor = true;
			// 
			// radioButton_Single
			// 
			this->radioButton_Single->AutoSize = true;
			this->radioButton_Single->Checked = true;
			this->radioButton_Single->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton_Single->Location = System::Drawing::Point(6, 25);
			this->radioButton_Single->Name = L"radioButton_Single";
			this->radioButton_Single->Size = System::Drawing::Size(61, 21);
			this->radioButton_Single->TabIndex = 0;
			this->radioButton_Single->TabStop = true;
			this->radioButton_Single->Text = L"Single";
			this->radioButton_Single->UseVisualStyleBackColor = true;
			// 
			// groupBox5
			// 
			this->groupBox5->Controls->Add(this->comboBox_CS);
			this->groupBox5->Controls->Add(this->radioButton_NC);
			this->groupBox5->Controls->Add(this->radioButton_C2);
			this->groupBox5->Controls->Add(this->radioButton_CS);
			this->groupBox5->Controls->Add(this->radioButton_C1);
			this->groupBox5->Location = System::Drawing::Point(554, 207);
			this->groupBox5->Name = L"groupBox5";
			this->groupBox5->Size = System::Drawing::Size(135, 107);
			this->groupBox5->TabIndex = 14;
			this->groupBox5->TabStop = false;
			this->groupBox5->Text = L"Target1";
			// 
			// comboBox_CS
			// 
			this->comboBox_CS->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 11.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->comboBox_CS->FormattingEnabled = true;
			this->comboBox_CS->Location = System::Drawing::Point(90, 51);
			this->comboBox_CS->Name = L"comboBox_CS";
			this->comboBox_CS->Size = System::Drawing::Size(34, 28);
			this->comboBox_CS->TabIndex = 16;
			this->comboBox_CS->Text = L"0";
			this->comboBox_CS->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_CS_SelectedIndexChanged);
			// 
			// radioButton_NC
			// 
			this->radioButton_NC->AutoSize = true;
			this->radioButton_NC->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton_NC->Location = System::Drawing::Point(6, 82);
			this->radioButton_NC->Name = L"radioButton_NC";
			this->radioButton_NC->Size = System::Drawing::Size(78, 21);
			this->radioButton_NC->TabIndex = 15;
			this->radioButton_NC->TabStop = true;
			this->radioButton_NC->Text = L"No Class";
			this->radioButton_NC->UseVisualStyleBackColor = true;
			this->radioButton_NC->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_NC_CheckedChanged);
			// 
			// radioButton_C2
			// 
			this->radioButton_C2->AutoSize = true;
			this->radioButton_C2->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton_C2->Location = System::Drawing::Point(6, 53);
			this->radioButton_C2->Name = L"radioButton_C2";
			this->radioButton_C2->Size = System::Drawing::Size(63, 21);
			this->radioButton_C2->TabIndex = 2;
			this->radioButton_C2->TabStop = true;
			this->radioButton_C2->Text = L"Class2";
			this->radioButton_C2->UseVisualStyleBackColor = true;
			this->radioButton_C2->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_C2_CheckedChanged);
			// 
			// radioButton_CS
			// 
			this->radioButton_CS->AutoSize = true;
			this->radioButton_CS->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton_CS->Location = System::Drawing::Point(69, 25);
			this->radioButton_CS->Name = L"radioButton_CS";
			this->radioButton_CS->Size = System::Drawing::Size(60, 21);
			this->radioButton_CS->TabIndex = 1;
			this->radioButton_CS->TabStop = true;
			this->radioButton_CS->Text = L"Select";
			this->radioButton_CS->UseVisualStyleBackColor = true;
			this->radioButton_CS->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_CS_CheckedChanged);
			// 
			// radioButton_C1
			// 
			this->radioButton_C1->AutoSize = true;
			this->radioButton_C1->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton_C1->Location = System::Drawing::Point(6, 25);
			this->radioButton_C1->Name = L"radioButton_C1";
			this->radioButton_C1->Size = System::Drawing::Size(63, 21);
			this->radioButton_C1->TabIndex = 0;
			this->radioButton_C1->TabStop = true;
			this->radioButton_C1->Text = L"Class1";
			this->radioButton_C1->UseVisualStyleBackColor = true;
			this->radioButton_C1->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_C1_CheckedChanged);
			// 
			// groupBox6
			// 
			this->groupBox6->Controls->Add(this->hScrollBar1);
			this->groupBox6->Controls->Add(this->textBox8);
			this->groupBox6->Controls->Add(this->label6);
			this->groupBox6->Controls->Add(this->textBox7);
			this->groupBox6->Controls->Add(this->label5);
			this->groupBox6->Controls->Add(this->textBox6);
			this->groupBox6->Controls->Add(this->label4);
			this->groupBox6->Controls->Add(this->textBox5);
			this->groupBox6->Controls->Add(this->label3);
			this->groupBox6->Controls->Add(this->groupBox7);
			this->groupBox6->Location = System::Drawing::Point(699, 138);
			this->groupBox6->Name = L"groupBox6";
			this->groupBox6->Size = System::Drawing::Size(135, 281);
			this->groupBox6->TabIndex = 15;
			this->groupBox6->TabStop = false;
			this->groupBox6->Text = L"Group Input";
			// 
			// hScrollBar1
			// 
			this->hScrollBar1->Location = System::Drawing::Point(12, 246);
			this->hScrollBar1->Name = L"hScrollBar1";
			this->hScrollBar1->Size = System::Drawing::Size(109, 22);
			this->hScrollBar1->TabIndex = 26;
			// 
			// textBox8
			// 
			this->textBox8->Location = System::Drawing::Point(83, 208);
			this->textBox8->Name = L"textBox8";
			this->textBox8->Size = System::Drawing::Size(39, 25);
			this->textBox8->TabIndex = 25;
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->label6->Location = System::Drawing::Point(3, 211);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(81, 15);
			this->label6->TabIndex = 24;
			this->label6->Text = L"Rotate Angle :";
			// 
			// textBox7
			// 
			this->textBox7->Location = System::Drawing::Point(83, 180);
			this->textBox7->Name = L"textBox7";
			this->textBox7->Size = System::Drawing::Size(39, 25);
			this->textBox7->TabIndex = 23;
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->label5->Location = System::Drawing::Point(3, 183);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(70, 15);
			this->label5->TabIndex = 22;
			this->label5->Text = L"Range of Y :";
			// 
			// textBox6
			// 
			this->textBox6->Location = System::Drawing::Point(83, 151);
			this->textBox6->Name = L"textBox6";
			this->textBox6->Size = System::Drawing::Size(39, 25);
			this->textBox6->TabIndex = 21;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->label4->Location = System::Drawing::Point(3, 154);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(70, 15);
			this->label4->TabIndex = 20;
			this->label4->Text = L"Range of X :";
			// 
			// textBox5
			// 
			this->textBox5->Location = System::Drawing::Point(82, 123);
			this->textBox5->Name = L"textBox5";
			this->textBox5->Size = System::Drawing::Size(39, 25);
			this->textBox5->TabIndex = 19;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->label3->Location = System::Drawing::Point(2, 126);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(65, 15);
			this->label3->TabIndex = 18;
			this->label3->Text = L"# of Point :";
			// 
			// groupBox7
			// 
			this->groupBox7->Controls->Add(this->radioButton10);
			this->groupBox7->Controls->Add(this->radioButton8);
			this->groupBox7->Location = System::Drawing::Point(8, 24);
			this->groupBox7->Name = L"groupBox7";
			this->groupBox7->Size = System::Drawing::Size(118, 83);
			this->groupBox7->TabIndex = 17;
			this->groupBox7->TabStop = false;
			this->groupBox7->Text = L"Distribution";
			this->groupBox7->Enter += gcnew System::EventHandler(this, &Form1::groupBox7_Enter);
			// 
			// radioButton10
			// 
			this->radioButton10->AutoSize = true;
			this->radioButton10->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton10->Location = System::Drawing::Point(11, 24);
			this->radioButton10->Name = L"radioButton10";
			this->radioButton10->Size = System::Drawing::Size(73, 21);
			this->radioButton10->TabIndex = 0;
			this->radioButton10->TabStop = true;
			this->radioButton10->Text = L"Uniform";
			this->radioButton10->UseVisualStyleBackColor = true;
			// 
			// radioButton8
			// 
			this->radioButton8->AutoSize = true;
			this->radioButton8->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->radioButton8->Location = System::Drawing::Point(11, 51);
			this->radioButton8->Name = L"radioButton8";
			this->radioButton8->Size = System::Drawing::Size(78, 21);
			this->radioButton8->TabIndex = 2;
			this->radioButton8->TabStop = true;
			this->radioButton8->Text = L"Gaussian";
			this->radioButton8->UseVisualStyleBackColor = true;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(557, 501);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(34, 17);
			this->label7->TabIndex = 16;
			this->label7->Text = L"File :";
			// 
			// textBox_Filename
			// 
			this->textBox_Filename->Location = System::Drawing::Point(601, 498);
			this->textBox_Filename->Name = L"textBox_Filename";
			this->textBox_Filename->Size = System::Drawing::Size(219, 25);
			this->textBox_Filename->TabIndex = 17;
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(565, 525);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(260, 196);
			this->richTextBox1->TabIndex = 18;
			this->richTextBox1->Text = L"";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(7, 137);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(25, 17);
			this->label8->TabIndex = 19;
			this->label8->Text = L"1.0";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(5, 343);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(25, 17);
			this->label9->TabIndex = 20;
			this->label9->Text = L"0.0";
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Location = System::Drawing::Point(4, 632);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(30, 17);
			this->label10->TabIndex = 21;
			this->label10->Text = L"-1.0";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(33, 652);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(30, 17);
			this->label11->TabIndex = 22;
			this->label11->Text = L"-1.0";
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Location = System::Drawing::Point(272, 652);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(25, 17);
			this->label12->TabIndex = 23;
			this->label12->Text = L"0.0";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(520, 652);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(25, 17);
			this->label13->TabIndex = 24;
			this->label13->Text = L"1.0";
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// groupBox8
			// 
			this->groupBox8->Controls->Add(this->comboBox_Run);
			this->groupBox8->Location = System::Drawing::Point(840, 75);
			this->groupBox8->Name = L"groupBox8";
			this->groupBox8->Size = System::Drawing::Size(137, 70);
			this->groupBox8->TabIndex = 25;
			this->groupBox8->TabStop = false;
			this->groupBox8->Text = L"Run Program";
			// 
			// comboBox_Run
			// 
			this->comboBox_Run->FormattingEnabled = true;
			this->comboBox_Run->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"Classification", L"Clustering", L"Regression"});
			this->comboBox_Run->Location = System::Drawing::Point(6, 33);
			this->comboBox_Run->Name = L"comboBox_Run";
			this->comboBox_Run->Size = System::Drawing::Size(121, 25);
			this->comboBox_Run->TabIndex = 0;
			this->comboBox_Run->Text = L"Classification";
			this->comboBox_Run->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_Run_SelectedIndexChanged);
			// 
			// groupBox9
			// 
			this->groupBox9->Controls->Add(this->comboBox_classify);
			this->groupBox9->Location = System::Drawing::Point(840, 162);
			this->groupBox9->Name = L"groupBox9";
			this->groupBox9->Size = System::Drawing::Size(137, 63);
			this->groupBox9->TabIndex = 26;
			this->groupBox9->TabStop = false;
			this->groupBox9->Text = L"Classificaion";
			// 
			// comboBox_classify
			// 
			this->comboBox_classify->FormattingEnabled = true;
			this->comboBox_classify->Items->AddRange(gcnew cli::array< System::Object^  >(7) {L"Bayes-Map", L"k-NN", L"Peroeptron", L"LVQ", 
				L"BPNN", L"LDA", L"SVM-SMO"});
			this->comboBox_classify->Location = System::Drawing::Point(6, 28);
			this->comboBox_classify->Name = L"comboBox_classify";
			this->comboBox_classify->Size = System::Drawing::Size(121, 25);
			this->comboBox_classify->TabIndex = 0;
			this->comboBox_classify->Text = L"Bayes-Map";
			// 
			// groupBox10
			// 
			this->groupBox10->Controls->Add(this->label14);
			this->groupBox10->Controls->Add(this->comboBox_clusters);
			this->groupBox10->Controls->Add(this->comboBox_clustering);
			this->groupBox10->Location = System::Drawing::Point(840, 243);
			this->groupBox10->Name = L"groupBox10";
			this->groupBox10->Size = System::Drawing::Size(146, 64);
			this->groupBox10->TabIndex = 27;
			this->groupBox10->TabStop = false;
			this->groupBox10->Text = L"Clustering";
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(85, 0);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(52, 17);
			this->label14->TabIndex = 28;
			this->label14->Text = L"clusters";
			// 
			// comboBox_clusters
			// 
			this->comboBox_clusters->FormattingEnabled = true;
			this->comboBox_clusters->Items->AddRange(gcnew cli::array< System::Object^  >(30) {L"2", L"3", L"4", L"5", L"6", L"7", L"8", 
				L"9", L"10", L"11", L"12", L"13", L"14", L"15", L"16", L"17", L"18", L"19", L"20", L"21", L"22", L"23", L"24", L"25", L"26", 
				L"27", L"28", L"29", L"30", L"31"});
			this->comboBox_clusters->Location = System::Drawing::Point(96, 24);
			this->comboBox_clusters->Name = L"comboBox_clusters";
			this->comboBox_clusters->Size = System::Drawing::Size(36, 25);
			this->comboBox_clusters->TabIndex = 1;
			this->comboBox_clusters->Text = L"2";
			// 
			// comboBox_clustering
			// 
			this->comboBox_clustering->FormattingEnabled = true;
			this->comboBox_clustering->Items->AddRange(gcnew cli::array< System::Object^  >(5) {L"k-Means", L"FCM", L"EM", L"GUCK", L"EGAC"});
			this->comboBox_clustering->Location = System::Drawing::Point(10, 24);
			this->comboBox_clustering->Name = L"comboBox_clustering";
			this->comboBox_clustering->Size = System::Drawing::Size(75, 25);
			this->comboBox_clustering->TabIndex = 0;
			this->comboBox_clustering->Text = L"k-Means";
			this->comboBox_clustering->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_clustering_SelectedIndexChanged);
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(85, 0);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(50, 17);
			this->label15->TabIndex = 28;
			this->label15->Text = L"degree";
			// 
			// groupBox11
			// 
			this->groupBox11->Controls->Add(this->label15);
			this->groupBox11->Controls->Add(this->comboBox_NL_degree);
			this->groupBox11->Controls->Add(this->comboBox_regression);
			this->groupBox11->Location = System::Drawing::Point(840, 318);
			this->groupBox11->Name = L"groupBox11";
			this->groupBox11->Size = System::Drawing::Size(146, 64);
			this->groupBox11->TabIndex = 28;
			this->groupBox11->TabStop = false;
			this->groupBox11->Text = L"Regression";
			// 
			// comboBox_NL_degree
			// 
			this->comboBox_NL_degree->FormattingEnabled = true;
			this->comboBox_NL_degree->Items->AddRange(gcnew cli::array< System::Object^  >(9) {L"2", L"3", L"4", L"5", L"6", L"7", L"8", 
				L"9", L"10"});
			this->comboBox_NL_degree->Location = System::Drawing::Point(96, 24);
			this->comboBox_NL_degree->Name = L"comboBox_NL_degree";
			this->comboBox_NL_degree->Size = System::Drawing::Size(36, 25);
			this->comboBox_NL_degree->TabIndex = 1;
			this->comboBox_NL_degree->Text = L"2";
			// 
			// comboBox_regression
			// 
			this->comboBox_regression->FormattingEnabled = true;
			this->comboBox_regression->Items->AddRange(gcnew cli::array< System::Object^  >(9) {L"Linear", L"Linear-Lin", L"Linear-Log10", 
				L"Linear-sat(1/r)", L"Nonlinear", L"k-NN", L"Perceptron", L"Logistic", L"BPNN"});
			this->comboBox_regression->Location = System::Drawing::Point(10, 24);
			this->comboBox_regression->Name = L"comboBox_regression";
			this->comboBox_regression->Size = System::Drawing::Size(75, 25);
			this->comboBox_regression->TabIndex = 0;
			this->comboBox_regression->Text = L"Linear";
			this->comboBox_regression->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_regression_SelectedIndexChanged);
			// 
			// groupBox12
			// 
			this->groupBox12->Controls->Add(this->groupBox13);
			this->groupBox12->Location = System::Drawing::Point(840, 399);
			this->groupBox12->Name = L"groupBox12";
			this->groupBox12->Size = System::Drawing::Size(150, 124);
			this->groupBox12->TabIndex = 29;
			this->groupBox12->TabStop = false;
			this->groupBox12->Text = L"General Parameter";
			// 
			// groupBox13
			// 
			this->groupBox13->Controls->Add(this->textBox_MaxIter);
			this->groupBox13->Controls->Add(this->label17);
			this->groupBox13->Controls->Add(this->textBox_delta);
			this->groupBox13->Controls->Add(this->label16);
			this->groupBox13->Location = System::Drawing::Point(10, 23);
			this->groupBox13->Name = L"groupBox13";
			this->groupBox13->Size = System::Drawing::Size(134, 92);
			this->groupBox13->TabIndex = 0;
			this->groupBox13->TabStop = false;
			this->groupBox13->Text = L"Stop Critaria";
			// 
			// textBox_MaxIter
			// 
			this->textBox_MaxIter->Location = System::Drawing::Point(68, 54);
			this->textBox_MaxIter->Name = L"textBox_MaxIter";
			this->textBox_MaxIter->Size = System::Drawing::Size(60, 25);
			this->textBox_MaxIter->TabIndex = 3;
			this->textBox_MaxIter->Text = L"1000";
			// 
			// label17
			// 
			this->label17->AutoSize = true;
			this->label17->Location = System::Drawing::Point(6, 54);
			this->label17->Name = L"label17";
			this->label17->Size = System::Drawing::Size(55, 17);
			this->label17->TabIndex = 2;
			this->label17->Text = L"MaxIter:";
			// 
			// textBox_delta
			// 
			this->textBox_delta->Location = System::Drawing::Point(52, 18);
			this->textBox_delta->Name = L"textBox_delta";
			this->textBox_delta->Size = System::Drawing::Size(76, 25);
			this->textBox_delta->TabIndex = 1;
			this->textBox_delta->Text = L"1.0e-8";
			// 
			// label16
			// 
			this->label16->AutoSize = true;
			this->label16->Location = System::Drawing::Point(6, 21);
			this->label16->Name = L"label16";
			this->label16->Size = System::Drawing::Size(40, 17);
			this->label16->TabIndex = 0;
			this->label16->Text = L"delta:";
			// 
			// groupBox14
			// 
			this->groupBox14->Controls->Add(this->textBox3);
			this->groupBox14->Controls->Add(this->label18);
			this->groupBox14->Controls->Add(this->checkBox_Unbiased);
			this->groupBox14->Location = System::Drawing::Point(840, 529);
			this->groupBox14->Name = L"groupBox14";
			this->groupBox14->Size = System::Drawing::Size(150, 100);
			this->groupBox14->TabIndex = 30;
			this->groupBox14->TabStop = false;
			this->groupBox14->Text = L"Bayes(Map...)";
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(94, 60);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(27, 25);
			this->textBox3->TabIndex = 2;
			this->textBox3->Text = L"1";
			// 
			// label18
			// 
			this->label18->AutoSize = true;
			this->label18->Location = System::Drawing::Point(6, 60);
			this->label18->Name = L"label18";
			this->label18->Size = System::Drawing::Size(88, 17);
			this->label18->TabIndex = 1;
			this->label18->Text = L"Ellipse Sigma:";
			// 
			// checkBox_Unbiased
			// 
			this->checkBox_Unbiased->AutoSize = true;
			this->checkBox_Unbiased->Checked = true;
			this->checkBox_Unbiased->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox_Unbiased->Location = System::Drawing::Point(16, 25);
			this->checkBox_Unbiased->Name = L"checkBox_Unbiased";
			this->checkBox_Unbiased->Size = System::Drawing::Size(82, 21);
			this->checkBox_Unbiased->TabIndex = 0;
			this->checkBox_Unbiased->Text = L"Unbiased";
			this->checkBox_Unbiased->UseVisualStyleBackColor = true;
			// 
			// groupBox15
			// 
			this->groupBox15->Controls->Add(this->comboBox_Kmeans_Option);
			this->groupBox15->Controls->Add(this->label20);
			this->groupBox15->Location = System::Drawing::Point(662, 75);
			this->groupBox15->Name = L"groupBox15";
			this->groupBox15->Size = System::Drawing::Size(159, 56);
			this->groupBox15->TabIndex = 31;
			this->groupBox15->TabStop = false;
			this->groupBox15->Text = L"K-Mean(RCM,EM)";
			// 
			// comboBox_Kmeans_Option
			// 
			this->comboBox_Kmeans_Option->FormattingEnabled = true;
			this->comboBox_Kmeans_Option->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"Original", L"Furthest Point", L"k-means++"});
			this->comboBox_Kmeans_Option->Location = System::Drawing::Point(60, 21);
			this->comboBox_Kmeans_Option->Name = L"comboBox_Kmeans_Option";
			this->comboBox_Kmeans_Option->Size = System::Drawing::Size(86, 25);
			this->comboBox_Kmeans_Option->TabIndex = 1;
			this->comboBox_Kmeans_Option->Text = L"Original";
			this->comboBox_Kmeans_Option->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_Kmeans_Option_SelectedIndexChanged);
			// 
			// label20
			// 
			this->label20->AutoSize = true;
			this->label20->Location = System::Drawing::Point(6, 21);
			this->label20->Name = L"label20";
			this->label20->Size = System::Drawing::Size(53, 17);
			this->label20->TabIndex = 0;
			this->label20->Text = L"Initial K:";
			// 
			// groupBox16
			// 
			this->groupBox16->Controls->Add(this->checkBox_ShowRange);
			this->groupBox16->Location = System::Drawing::Point(554, 321);
			this->groupBox16->Name = L"groupBox16";
			this->groupBox16->Size = System::Drawing::Size(135, 46);
			this->groupBox16->TabIndex = 32;
			this->groupBox16->TabStop = false;
			this->groupBox16->Text = L"Clurstered Range";
			// 
			// checkBox_ShowRange
			// 
			this->checkBox_ShowRange->AutoSize = true;
			this->checkBox_ShowRange->Checked = true;
			this->checkBox_ShowRange->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox_ShowRange->Location = System::Drawing::Point(25, 19);
			this->checkBox_ShowRange->Name = L"checkBox_ShowRange";
			this->checkBox_ShowRange->Size = System::Drawing::Size(58, 21);
			this->checkBox_ShowRange->TabIndex = 0;
			this->checkBox_ShowRange->Text = L"Show";
			this->checkBox_ShowRange->UseVisualStyleBackColor = true;
			// 
			// groupBox17
			// 
			this->groupBox17->Controls->Add(this->label22);
			this->groupBox17->Controls->Add(this->textBox1);
			this->groupBox17->Controls->Add(this->label21);
			this->groupBox17->Controls->Add(this->label19);
			this->groupBox17->Controls->Add(this->comboBox_Weight);
			this->groupBox17->Controls->Add(this->comboBox_kNN);
			this->groupBox17->Location = System::Drawing::Point(983, 75);
			this->groupBox17->Name = L"groupBox17";
			this->groupBox17->Size = System::Drawing::Size(167, 88);
			this->groupBox17->TabIndex = 33;
			this->groupBox17->TabStop = false;
			this->groupBox17->Text = L"k-NN";
			// 
			// label22
			// 
			this->label22->AutoSize = true;
			this->label22->Location = System::Drawing::Point(6, 60);
			this->label22->Name = L"label22";
			this->label22->Size = System::Drawing::Size(64, 17);
			this->label22->TabIndex = 5;
			this->label22->Text = L"weighted:";
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(116, 24);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(33, 25);
			this->textBox1->TabIndex = 4;
			this->textBox1->Text = L"0.01";
			// 
			// label21
			// 
			this->label21->AutoSize = true;
			this->label21->Location = System::Drawing::Point(81, 27);
			this->label21->Name = L"label21";
			this->label21->Size = System::Drawing::Size(33, 17);
			this->label21->TabIndex = 3;
			this->label21->Text = L"s^2:";
			this->label21->Click += gcnew System::EventHandler(this, &Form1::label21_Click);
			// 
			// label19
			// 
			this->label19->AutoSize = true;
			this->label19->Location = System::Drawing::Point(7, 29);
			this->label19->Name = L"label19";
			this->label19->Size = System::Drawing::Size(17, 17);
			this->label19->TabIndex = 2;
			this->label19->Text = L"k:";
			// 
			// comboBox_Weight
			// 
			this->comboBox_Weight->FormattingEnabled = true;
			this->comboBox_Weight->Items->AddRange(gcnew cli::array< System::Object^  >(3) {L"Average", L"1/Dist", L"RBF"});
			this->comboBox_Weight->Location = System::Drawing::Point(76, 56);
			this->comboBox_Weight->Name = L"comboBox_Weight";
			this->comboBox_Weight->Size = System::Drawing::Size(85, 25);
			this->comboBox_Weight->TabIndex = 1;
			this->comboBox_Weight->Text = L"Average";
			// 
			// comboBox_kNN
			// 
			this->comboBox_kNN->FormattingEnabled = true;
			this->comboBox_kNN->Items->AddRange(gcnew cli::array< System::Object^  >(30) {L"1", L"3", L"5", L"7", L"9", L"11", L"13", 
				L"15", L"17", L"19", L"21", L"23", L"25", L"27", L"29", L"31", L"33", L"35", L"37", L"39", L"41", L"43", L"45", L"47", L"49", 
				L"51", L"53", L"55", L"57", L"59"});
			this->comboBox_kNN->Location = System::Drawing::Point(24, 25);
			this->comboBox_kNN->Name = L"comboBox_kNN";
			this->comboBox_kNN->Size = System::Drawing::Size(46, 25);
			this->comboBox_kNN->TabIndex = 0;
			this->comboBox_kNN->Text = L"1";
			this->comboBox_kNN->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_kNN_SelectedIndexChanged);
			// 
			// groupBox18
			// 
			this->groupBox18->Controls->Add(this->comboBox_P_Function);
			this->groupBox18->Location = System::Drawing::Point(6, 202);
			this->groupBox18->Name = L"groupBox18";
			this->groupBox18->Size = System::Drawing::Size(144, 53);
			this->groupBox18->TabIndex = 34;
			this->groupBox18->TabStop = false;
			this->groupBox18->Text = L"Perceptron";
			// 
			// comboBox_P_Function
			// 
			this->comboBox_P_Function->FormattingEnabled = true;
			this->comboBox_P_Function->Items->AddRange(gcnew cli::array< System::Object^  >(4) {L"hardlims", L"linear", L"sigmoid", L"tanh()"});
			this->comboBox_P_Function->Location = System::Drawing::Point(10, 22);
			this->comboBox_P_Function->Name = L"comboBox_P_Function";
			this->comboBox_P_Function->Size = System::Drawing::Size(121, 25);
			this->comboBox_P_Function->TabIndex = 0;
			this->comboBox_P_Function->Text = L"hardlims";
			// 
			// groupBox19
			// 
			this->groupBox19->Controls->Add(this->comboBox2);
			this->groupBox19->Controls->Add(this->textBox2);
			this->groupBox19->Controls->Add(this->label23);
			this->groupBox19->Location = System::Drawing::Point(6, 310);
			this->groupBox19->Name = L"groupBox19";
			this->groupBox19->Size = System::Drawing::Size(144, 79);
			this->groupBox19->TabIndex = 35;
			this->groupBox19->TabStop = false;
			this->groupBox19->Text = L"BP";
			// 
			// comboBox2
			// 
			this->comboBox2->FormattingEnabled = true;
			this->comboBox2->Location = System::Drawing::Point(42, 16);
			this->comboBox2->Name = L"comboBox2";
			this->comboBox2->Size = System::Drawing::Size(86, 25);
			this->comboBox2->TabIndex = 2;
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(71, 47);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(42, 25);
			this->textBox2->TabIndex = 1;
			// 
			// label23
			// 
			this->label23->AutoSize = true;
			this->label23->Location = System::Drawing::Point(12, 50);
			this->label23->Name = L"label23";
			this->label23->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->label23->Size = System::Drawing::Size(53, 17);
			this->label23->TabIndex = 0;
			this->label23->Text = L"Hidden:";
			// 
			// groupBox20
			// 
			this->groupBox20->Controls->Add(this->textBox_Epsilon);
			this->groupBox20->Controls->Add(this->label28);
			this->groupBox20->Controls->Add(this->groupBox21);
			this->groupBox20->Controls->Add(this->groupBox19);
			this->groupBox20->Controls->Add(this->groupBox18);
			this->groupBox20->Location = System::Drawing::Point(1007, 186);
			this->groupBox20->Name = L"groupBox20";
			this->groupBox20->Size = System::Drawing::Size(175, 443);
			this->groupBox20->TabIndex = 36;
			this->groupBox20->TabStop = false;
			this->groupBox20->Text = L"Neural Networks";
			// 
			// textBox_Epsilon
			// 
			this->textBox_Epsilon->Location = System::Drawing::Point(70, 276);
			this->textBox_Epsilon->Name = L"textBox_Epsilon";
			this->textBox_Epsilon->Size = System::Drawing::Size(80, 25);
			this->textBox_Epsilon->TabIndex = 37;
			this->textBox_Epsilon->Text = L"0.001";
			// 
			// label28
			// 
			this->label28->AutoSize = true;
			this->label28->Location = System::Drawing::Point(10, 276);
			this->label28->Name = L"label28";
			this->label28->Size = System::Drawing::Size(53, 17);
			this->label28->TabIndex = 36;
			this->label28->Text = L"Epsilon:";
			// 
			// groupBox21
			// 
			this->groupBox21->Controls->Add(this->label27);
			this->groupBox21->Controls->Add(this->label26);
			this->groupBox21->Controls->Add(this->textBox_bias);
			this->groupBox21->Controls->Add(this->label25);
			this->groupBox21->Controls->Add(this->textBox4);
			this->groupBox21->Controls->Add(this->label24);
			this->groupBox21->Controls->Add(this->textBox_initail);
			this->groupBox21->Controls->Add(this->comboBox1);
			this->groupBox21->Location = System::Drawing::Point(6, 21);
			this->groupBox21->Name = L"groupBox21";
			this->groupBox21->Size = System::Drawing::Size(144, 175);
			this->groupBox21->TabIndex = 0;
			this->groupBox21->TabStop = false;
			this->groupBox21->Text = L"Learning Rate";
			// 
			// label27
			// 
			this->label27->AutoSize = true;
			this->label27->Location = System::Drawing::Point(0, 131);
			this->label27->Name = L"label27";
			this->label27->Size = System::Drawing::Size(53, 17);
			this->label27->TabIndex = 7;
			this->label27->Text = L"b/(b+n)";
			// 
			// label26
			// 
			this->label26->AutoSize = true;
			this->label26->Location = System::Drawing::Point(31, 114);
			this->label26->Name = L"label26";
			this->label26->Size = System::Drawing::Size(19, 17);
			this->label26->TabIndex = 6;
			this->label26->Text = L"b:";
			// 
			// textBox_bias
			// 
			this->textBox_bias->Location = System::Drawing::Point(54, 113);
			this->textBox_bias->Multiline = true;
			this->textBox_bias->Name = L"textBox_bias";
			this->textBox_bias->Size = System::Drawing::Size(60, 40);
			this->textBox_bias->TabIndex = 5;
			this->textBox_bias->Text = L"0.5";
			// 
			// label25
			// 
			this->label25->AutoSize = true;
			this->label25->Location = System::Drawing::Point(3, 90);
			this->label25->Name = L"label25";
			this->label25->Size = System::Drawing::Size(47, 17);
			this->label25->TabIndex = 4;
			this->label25->Text = L"DF/M :";
			// 
			// textBox4
			// 
			this->textBox4->Location = System::Drawing::Point(54, 84);
			this->textBox4->Name = L"textBox4";
			this->textBox4->Size = System::Drawing::Size(60, 25);
			this->textBox4->TabIndex = 3;
			this->textBox4->Text = L"0.95";
			// 
			// label24
			// 
			this->label24->AutoSize = true;
			this->label24->Location = System::Drawing::Point(3, 57);
			this->label24->Name = L"label24";
			this->label24->Size = System::Drawing::Size(41, 17);
			this->label24->TabIndex = 2;
			this->label24->Text = L"initail:";
			// 
			// textBox_initail
			// 
			this->textBox_initail->Location = System::Drawing::Point(54, 51);
			this->textBox_initail->Name = L"textBox_initail";
			this->textBox_initail->Size = System::Drawing::Size(60, 25);
			this->textBox_initail->TabIndex = 1;
			this->textBox_initail->Text = L"0.5";
			// 
			// comboBox1
			// 
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Location = System::Drawing::Point(6, 21);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(121, 25);
			this->comboBox1->TabIndex = 0;
			this->comboBox1->Text = L"Specified";
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(7, 17);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1194, 733);
			this->Controls->Add(this->groupBox20);
			this->Controls->Add(this->groupBox17);
			this->Controls->Add(this->groupBox16);
			this->Controls->Add(this->groupBox15);
			this->Controls->Add(this->groupBox14);
			this->Controls->Add(this->groupBox12);
			this->Controls->Add(this->groupBox11);
			this->Controls->Add(this->groupBox10);
			this->Controls->Add(this->groupBox9);
			this->Controls->Add(this->groupBox8);
			this->Controls->Add(this->label13);
			this->Controls->Add(this->label12);
			this->Controls->Add(this->label11);
			this->Controls->Add(this->label10);
			this->Controls->Add(this->label9);
			this->Controls->Add(this->label8);
			this->Controls->Add(this->richTextBox1);
			this->Controls->Add(this->textBox_Filename);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->groupBox6);
			this->Controls->Add(this->groupBox5);
			this->Controls->Add(this->groupBox4);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->Run);
			this->Controls->Add(this->button_clear);
			this->Controls->Add(this->groupBox3);
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->textBox_Y);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->textBox_X);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->checkBox1);
			this->Controls->Add(this->toolStrip1);
			this->Controls->Add(this->menuStrip1);
			this->Font = (gcnew System::Drawing::Font(L"Segoe UI Emoji", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->MainMenuStrip = this->menuStrip1;
			this->Margin = System::Windows::Forms::Padding(3, 5, 3, 5);
			this->Name = L"Form1";
			this->Text = L"MachineLearning(U10216035)";
			this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->toolStrip1->ResumeLayout(false);
			this->toolStrip1->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->pictureBox1))->EndInit();
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			this->groupBox5->ResumeLayout(false);
			this->groupBox5->PerformLayout();
			this->groupBox6->ResumeLayout(false);
			this->groupBox6->PerformLayout();
			this->groupBox7->ResumeLayout(false);
			this->groupBox7->PerformLayout();
			this->groupBox8->ResumeLayout(false);
			this->groupBox9->ResumeLayout(false);
			this->groupBox10->ResumeLayout(false);
			this->groupBox10->PerformLayout();
			this->groupBox11->ResumeLayout(false);
			this->groupBox11->PerformLayout();
			this->groupBox12->ResumeLayout(false);
			this->groupBox13->ResumeLayout(false);
			this->groupBox13->PerformLayout();
			this->groupBox14->ResumeLayout(false);
			this->groupBox14->PerformLayout();
			this->groupBox15->ResumeLayout(false);
			this->groupBox15->PerformLayout();
			this->groupBox16->ResumeLayout(false);
			this->groupBox16->PerformLayout();
			this->groupBox17->ResumeLayout(false);
			this->groupBox17->PerformLayout();
			this->groupBox18->ResumeLayout(false);
			this->groupBox19->ResumeLayout(false);
			this->groupBox19->PerformLayout();
			this->groupBox20->ResumeLayout(false);
			this->groupBox20->PerformLayout();
			this->groupBox21->ResumeLayout(false);
			this->groupBox21->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

private: System::Void Form1_Load(System::Object^  sender, System::EventArgs^  e) {
		myBitmap= gcnew Bitmap(512, 512, PixelFormat::Format24bppRgb);
		g = Graphics::FromImage(myBitmap);
		button_Clear_Click(sender, e);
		Pi = 4.0 * atan(1.0);
		imW= pictureBox1->Width;
		imH= pictureBox1->Height;
		CenterX=0.5*imW; //Center X®y¼Ð
		CenterY=0.5*imH; //Center Y®y¼Ð
		comboBox_psize->SelectedIndex=1; //PointSize=2¡A¹w³]¸ê®ÆÂI¤j¤p¡C
		PointSize=(comboBox_psize->SelectedIndex+1)*5;
		PointSize1=PointSize+2; //¸ê®ÆÂI¹Bºâ§¹µ²ªG¤j¤p
		PointSize2=PointSize+4; //Class or Cluster ¤¤¤ß¤j¤p
		radioButton_C1->Checked=true; //ClassKind=1; Color=Red
		ClassKind=1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
		//comboBox_CSItems initializing, i.e. Class(Target) selection.
		totalCTestData = imW*imH;
		MaxKNN = 1;
		totalRTestData = imW;
		

		for (int i=0;i<10;i++)
			comboBox_CS->Items->Add(Convert::ToString(i));
		comboBox_CS->SelectedIndex=0;
		comboBox_CS->Enabled=false;
		HandFlag=true; //HandFlag=true®É¡APictureBox1_MouseClick¥i¿é¤JData Points¡C¤Ï¤§¡A«h¤£¯à¡C
		//Group Input
		//srand( (unsigned)time(NULL) );
		//Distribution=1; //Distribution=1 --> Gaussian , Distribution=0 --> Uniform
		//NumberOfPoint=Convert::ToInt32(textBox_Num_p->Text); //Number Of Point per Click
		//RangeX=Convert::ToInt32(textBox_R_X->Text); //Range of X
		//RangeY=Convert::ToInt32(textBox_R_Y->Text); //Range of Y

		//Run Program
		comboBox_Run->SelectedIndex=0; //Run Classification--1:Clustering--2:Regression
		comboBox_clustering->Enabled=false; //Disable Clustering
		comboBox_clusters->Enabled=false; //Disable Clustering
		comboBox_regression->Enabled=false; //Disable Regression
		//classification Method
		comboBox_classify->SelectedIndex=0;

		//Regression
		comboBox_NL_degree->Enabled=false; //Nonlinear Regression only use.

		MaxSizeOfData = Convert::ToInt32(textBox_MaxSize->Text); //MaxSizeOf Input Data
		InputData= new pData[MaxSizeOfData];
		NewPublicVariables(MaxSizeOfData);

		//clustering Method
		STOPFlag=true;
		comboBox_clustering->SelectedIndex=0;
		comboBox_Kmeans_Option->SelectedIndex=0;
		for (int i=0;i<30;i++)
			comboBox_clusters->Items->Add(Convert::ToString(i+2));
		comboBox_clusters->SelectedIndex=0;
		NumOfCluster= comboBox_clusters->SelectedIndex+2;

		 }
private: System::Void button_Clear_Click(System::Object^  sender, System::EventArgs^  e) {
			 clearImageToolStripMenuItem_Click(sender, e);
			NumberOfData=0;
			textBox_datasize->Text = "0";
			Filename1="";
			textBox_Filename->Text ="";
			richTextBox1->Clear();
		 }
private: System::Void pictureBox1_Click(System::Object^  sender, System::EventArgs^  e) {
			 

		 }
private: System::Void pictureBox1_MouseMove(System::Object^ sender,System::Windows::Forms::MouseEventArgs^ e) {
		//textBox_X->Text =Convert::ToString((e->X-256.0)/256.0);
		//textBox_Y->Text =Convert::ToString((256.0-e->Y)/256.0);
		textBox_X->Text =Convert::ToString((e->X-CenterX)/CenterX);
		textBox_Y->Text =Convert::ToString((CenterY-e->Y)/CenterY);
 }
private: System::Void pictureBox1_MouseDown(System::Object^ sender,System::Windows::Forms::MouseEventArgs^ e) {
		 X_Cur=e->X;
		 Y_Cur=e->Y;
}

private: System::Void pictureBox1_MouseClick(System::Object^ sender,System::Windows::Forms::MouseEventArgs^ e) {
		double X_tmp=0.0,Y_tmp=0.0;
		if (HandFlag) { //HandFlag=true®É¡APictureBox1_MouseClick¥i¿é¤JData Points¡C¤Ï¤§¡A«h¤£¯à¡C
			if (radioButton_Single->Checked) {
				//X_tmp=(double)(X_Cur-256.0)/256.0;
				//Y_tmp=(double)(256.0-Y_Cur)/256.0;
				X_tmp=(double)(X_Cur-CenterX)/CenterX;
				Y_tmp=(double)(CenterY-Y_Cur)/CenterY;
				InputData[NumberOfData].X=X_tmp;
				InputData[NumberOfData].Y=Y_tmp;
				if (radioButton_C1->Checked){
					InputData[NumberOfData].ClassKind=1; //Red Color
					NumClass1++;
				}//if (radioButton_C1->Checked)
				else if (radioButton_C2->Checked){
					InputData[NumberOfData].ClassKind=-1; //Blue Color
					NumClass2++;
				}//else if (radioButton_C2->Checked)
				else if (radioButton_NC->Checked){
					InputData[NumberOfData].ClassKind=-2; //Black Color
					NumClass2++;
				}//else if (radioButton_NC->Checked)
				else{
					InputData[NumberOfData].ClassKind=comboBox_CS->SelectedIndex; //Select Class Color
					NumNoclass++;
				}//else
				bshDraw=ClassToColor(InputData[NumberOfData].ClassKind);
				g->FillEllipse(bshDraw, X_Cur-PointSize/2, Y_Cur-PointSize/2, PointSize, PointSize);
				NumberOfData++;
				textBox_datasize->Text = Convert::ToString(NumberOfData);
			}//if single input
			else { //group input
				MessageBox::Show("«Øºc¤¤¡A½Ð­@¤ßµ¥­Ô");
			}//else Group input
		}//if (HandFlag)
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
}

private: System::Void toolStripButton2_Click(System::Object^  sender, System::EventArgs^  e) {
			if ( String::IsNullOrEmpty(Filename1) )
				saveAsToolStripMenuItem_Click(sender, e);
			else
				richTextBox1->SaveFile(Filename1);
		 }
private: System::Void radioButton_C1_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if (radioButton_C1->Checked){
				ClassKind=1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
			}//if	
		 }
private: System::Void toolStripButton3_Click(System::Object^  sender, System::EventArgs^  e) {
			delete g;
			delete [] InputData;
			//DeletePublicVariables(MaxSizeOfData);
			Application::Exit();
		 }
private: System::Void toolStripButton6_Click(System::Object^  sender, System::EventArgs^  e) {
			if (NumberOfData>0) {
			richTextBox1->Clear();
			for (int i=0; i<NumberOfData; i++) {
				String^ line = Convert::ToString(InputData[i].X) + "\t" + Convert::ToString(InputData[i].Y) + "\t"+ Convert::ToString(InputData[i].ClassKind)+ "\n";
				richTextBox1->AppendText(line);
			}//for i
			saveFileDialog1->Filter="*.TXT|*.txt|*.DAT|*.dat|All Files|*.*";
			//saveFileDialog1->DefaultExt="dat";
			saveFileDialog1->DefaultExt = "txt";
			if (saveFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK && saveFileDialog1->FileName->Length > 0) {
				richTextBox1->SaveFile(saveFileDialog1->FileName);
				Filename1 = saveFileDialog1->FileName;
				textBox_Filename->Text = saveFileDialog1->FileName;
			}//if saveFileDialog1
			}//if (NumberOfData>0)
		 }
private: System::Void toolStripButton4_Click(System::Object^  sender, System::EventArgs^  e) {
			 bool openfile = true;
			if (richTextBox1->Modified) { //§PÂ_ richTextBox1¬O§_¦³¸g¹L¥ô¦ó½s¿è
				bool openfile = false;
				if (MessageBox::Show("¥¼¦sÀÉ!¬O§_Ä~Äò", "½T»{µøµ¡", MessageBoxButtons::YesNo, MessageBoxIcon::Question)==System::Windows::Forms::DialogResult::Yes)
				openfile = true;
			}//if (richTextBox1->Modified)
			if (openfile) {
				openFileDialog1->Filter="*.TXT|*.txt|*.DAT|*.dat|All Files|*.*";
				if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
					button_Clear_Click(sender, e);
					richTextBox1->LoadFile(openFileDialog1->FileName);
					Filename1 = openFileDialog1->FileName;
					textBox_Filename->Text = openFileDialog1->SafeFileName;
					NumberOfData=0;
					//char* Token =(char*)(void*)Marshal::StringToHGlobalAnsi(richTextBox1->Text);
					//char* Token =(char*)(void*)Marshal::StringToHGlobalAnsi(richTextBox1->Text);
					char* Token=(char*)Marshal::StringToHGlobalAnsi(richTextBox1->Text).ToPointer();
					char* sptoken = strtok(Token," \t\n");
					while (sptoken != NULL) {
						InputData[NumberOfData].X= atof(sptoken);
						sptoken = strtok(NULL," \t\n");
						InputData[NumberOfData].Y= atof(sptoken);
						sptoken = strtok(NULL," \t\n");
						InputData[NumberOfData].ClassKind= atoi(sptoken);
						sptoken = strtok(NULL," \t\n");
						NumberOfData++;
					}//while
					textBox_datasize->Text = Convert::ToString(NumberOfData);
					showDataToolStripMenuItem_Click(sender, e); // // Show data
					//Backup ClassKindfor clustering
					for(int j = 0;j <NumberOfData; j++)
						BackupClassKind[j] = InputData[j].ClassKind;
				}//if openFileDialog1
			}//if (openfile)
		 }
private: System::Void textBox_MaxSize_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			 MaxSizeOfData=Convert::ToInt32(textBox_MaxSize->Text);
			if (MaxSizeOfData<1000){
				MaxSizeOfData=1000;
				textBox_MaxSize->Text=Convert::ToString(MaxSizeOfData);
			}//if
			else {
				//DeletePublicVariables(MaxSizeOfData);
				delete [] InputData;
				//NewPublicVariables(MaxSizeOfData);
				InputData = new pData[MaxSizeOfData];
			}//else
			button_Clear_Click(sender, e);
		 }
private: System::Void checkBox1_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void saveAsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 toolStripButton6_Click(sender,e);
		 }
private: System::Void openToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 toolStripButton4_Click(sender,e);
		 }
private: System::Void saveToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 toolStripButton2_Click(sender,e);
		 }
private: System::Void exitToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 toolStripButton3_Click(sender,e);
		 }
private: System::Void clearImageToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			  g->Clear(Color::White); //Paint over the image area in white.
			//bshDraw = gcnew SolidBrush(Color::White); //²M°£µe¥¬¥t¤@ºØ¤èªk
			//g->FillRectangle(bshDraw, 0, 0, 512, 512);
			//Draw
			penDraw = gcnew Pen(Color::Black, 1); //µeµ§¬O¶Â¦âªº, µe®Ø
			//g->DrawLine(penDraw, 0, 0, 0, 511);
			//g->DrawLine(penDraw, 0, 0, 511, 0);
			//g->DrawLine(penDraw, 0, 511, 511, 511);
			//g->DrawLine(penDraw, 511, 0, 511, 511);
			g->DrawLine(penDraw, 0, 0, 0, pictureBox1->Height-1);
			g->DrawLine(penDraw, 0, 0, pictureBox1->Width-1, 0);
			g->DrawLine(penDraw, 0, pictureBox1->Height-1, pictureBox1->Width-1, pictureBox1->Height-1);
			g->DrawLine(penDraw, pictureBox1->Width-1, 0, pictureBox1->Width-1, pictureBox1->Height-1);
			pictureBox1->Image = myBitmap;
			pictureBox1->Refresh();
		 }
private: System::Void dataEditToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void showDataToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {

			 clearImageToolStripMenuItem_Click(sender, e);
			//DrawData
			for (int i=0; i<NumberOfData; i++) {
				//X_Cur=(int)(InputData[i].X*256+256);
				//Y_Cur=(int)(256-InputData[i].Y*256);
				X_Cur=(int)(InputData[i].X*CenterX+CenterX);
				Y_Cur=(int)(CenterY-InputData[i].Y*CenterY);
				bshDraw=ClassToColor(InputData[i].ClassKind);
				g->FillEllipse(bshDraw, X_Cur-(PointSize/2), Y_Cur-(PointSize/2), PointSize, PointSize);
			}// for
			pictureBox1->Image = myBitmap;
			pictureBox1->Refresh();
		 }
private: System::Void radioButton_C2_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if (radioButton_C2->Checked){
				ClassKind=-1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
			}//if
		 }
private: System::Void radioButton_NC_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 if (radioButton_NC->Checked){
				ClassKind=-2; //ClassKind=-2; Color=Black
			}//if
		 }
private: System::Void radioButton_CS_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			 comboBox_CS->Enabled=radioButton_CS->Checked;
			if (radioButton_CS->Checked){
				ClassKind=comboBox_CS->SelectedIndex;
			}//if
		 }
private: System::Void comboBox_CS_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			if (radioButton_CS->Checked){
				ClassKind=comboBox_CS->SelectedIndex;
			}//if
		 }
private: System::Void comboBox_psize_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 PointSize=(comboBox_psize->SelectedIndex+1)*5;
			PointSize1=PointSize+2;
			PointSize2=PointSize+4;
		 }
private: System::Void run_Click(System::Object^  sender, System::EventArgs^  e) {
			 MethodCodeValue= 0;
			int methodbased= 20;
			
			switch (comboBox_Run->SelectedIndex) {
			case 0: //Classification
				MethodCodeValue= (comboBox_Run->SelectedIndex)*methodbased+ comboBox_classify->SelectedIndex;
				break;
			case 1: //Clustering
				MethodCodeValue= (comboBox_Run->SelectedIndex)*methodbased+ comboBox_clustering->SelectedIndex;
				break;
			case 2: //Regression
				MethodCodeValue= (comboBox_Run->SelectedIndex)*methodbased+ comboBox_regression->SelectedIndex;
				break;
			default:
				MessageBox::Show("µL«Øºc¦¹Ãþ§O¤èªk!");
			}//switch

			switch (MethodCodeValue) {
			case 0: //Classification--Bayes-MAP
				BayesMAP();
				showContourToolStripMenuItem_Click(sender, e);
				showResultToolStripMenuItem_Click(sender, e);
				showMeansToolStripMenuItem_Click(sender, e);
				break;
			case 1: //k-NN
				kNNs= comboBox_kNN->SelectedIndex*2+1;
				if (MaxKNN>0) {
					if (!CreatekNNFlag) Create_kNN_Contour_Table();
						showContourToolStripMenuItem_Click(sender, e);
						showResultToolStripMenuItem_Click(sender, e);
					}//if
				break;
			case 2: //Perceptron Classification
				comboBox_P_Function->SelectedIndex=0; //«ü©wTransfer Function = hardlims()
				Perceptron_Ctrain(); //Training Weights
				showContourToolStripMenuItem_Click(sender, e);
				showResultToolStripMenuItem_Click(sender, e);
				break;
			case 20: //Clustering--K-Means
				NumOfClusters=comboBox_clusters->SelectedIndex+2;
				K_Means(NumOfClusters);
				clearImageToolStripMenuItem_Click(sender, e);
				showClusteredToolStripMenuItem_Click(sender, e);
				showClusterCenterToolStripMenuItem_Click(sender, e);
				break;
			case 21: //Clustering¡XFuzzy C-Means
				NumOfClusters=comboBox_clusters->SelectedIndex+2;
				FCM(NumOfClusters);
				clearImageToolStripMenuItem_Click(sender, e);
				showClusteredToolStripMenuItem_Click(sender, e);
				showClusterCenterToolStripMenuItem_Click(sender, e);
				break;
			case 40:
				LinearRegression();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 41: //Regression--Linear-ln() == log e
				LinearRegressionLn();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 44: //Nonlinear Regression--Degree == NLdegree
				//comboBox_NL_degree->SelectedIndex = 0;
				NLdegree= comboBox_NL_degree->SelectedIndex+2;
				A=new double*[NLdegree+1];
				for (int i=0; i<NLdegree+1; i++)
					A[i]= new double[NLdegree+1];
				B=new double[NLdegree+1];
				NLcoef=new double[NLdegree+1];
				NonlinearRegression(NLdegree);
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				//delete
				for (int i=0; i<NLdegree+1; i++)
					delete [] A[i];
				delete [] A;
				delete [] B;
				delete [] NLcoef;
				break;
			case 45: // Regression¡Xk-NN
				kNNs= comboBox_kNN->SelectedIndex*2+1;
				if (!BuiltkNNFlag)
					BuildAllkNNRegTable();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 46: //Perceptron Regression
				if (comboBox_P_Function->SelectedIndex==0) //­YTransfer Function = hardlims()¡A«h«ü©w¦¨linear()
					comboBox_P_Function->SelectedIndex=1; //¦]Transfer Function¶·¬°³sÄò¨ç¼Æ¤~¥i·L¤À¡C
				tF = comboBox_P_Function->SelectedIndex;
				Perceptron_RTrain(tF); //Training Weights
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			default:
				MessageBox::Show("µL«Øºc¦¹Ãþ§O¤èªk!");
			}//switch (MethodCodeValue)
		 }
private: System::Void showResultToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 	double Output_yi;
		Brush^ errBrush= gcnew SolidBrush(Color::DeepPink); //Classified error color
		//Draw Data
		for (int i = 0; i<NumberOfData; i++) {
			X_Cur= (int)(InputData[i].X*CenterX+ CenterX); //CenterX=256.0
			Y_Cur= (int)(CenterY-InputData[i].Y*CenterY); //CenterY=256.0
			switch (MethodCodeValue) {
			case 0: //Bayesian-MAP
				Output_yi= PClass1*PxyClass1[i] -PClass2*PxyClass2[i];
				if (Sgn((double)InputData[i].ClassKind) == Sgn(Output_yi)) {
					bshDraw= ClassToColor(InputData[i].ClassKind);
					g->FillEllipse(bshDraw, X_Cur-PointSize1 / 2, Y_Cur-PointSize1 / 2, PointSize1, PointSize1);
				}//if (Sgn((double) InputData[i].ClassKind)== Sgn(Output_yi))
				else {
					g->FillEllipse(errBrush, X_Cur-PointSize1 / 2, Y_Cur-PointSize1 / 2, PointSize1, PointSize1);
				}//else
				break;
			case 1: //K-NN
				bshDraw = ClassToColor(InputData[i].ClassKind);
				g->FillEllipse(bshDraw, X_Cur - PointSize1 / 2, Y_Cur - PointSize1 / 2, PointSize1, PointSize1);
				break;
			case 2: // Perceptron Classification
				Output_yi=(double) InputData[i].ClassKind*PerceptronClassify(InputData[i]);
				if (Sgn(Output_yi)>0) {
				bshDraw=ClassToColor(InputData[i].ClassKind);
				g->FillEllipse(bshDraw, X_Cur-PointSize1/2, Y_Cur-PointSize1/2, PointSize1, PointSize1);
				}//if
				else {
				g->FillEllipse(errBrush, X_Cur-PointSize1/2, Y_Cur-PointSize1/2, PointSize1, PointSize1);
				}//else
				break;
			default:
				MessageBox::Show("µL«Øºc¦¹Ãþ§O¤èªk!");
			}//switch
		}//for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
		 }
private: System::Void showContourToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			double Output_yi;
			
			pData Sample;
			unsigned char blockcolor=0, LightLevel, ColorScale;
			int Lighttmp;
			int i;
			//DrawData
			// Lock the bitmap's bits.
			Rectangle rect= Rectangle(0, 0, imW, imH);
			BitmapData^ bmpData= myBitmap->LockBits(rect, ImageLockMode::ReadWrite, myBitmap->PixelFormat);
			int ByteOfSkip= bmpData->Stride -bmpData->Width * 3;//­pºâ¨C¦æ«á­±´X­ÓPadding bytes , ¥þ±m¼v¹³
			unsigned char* p = (unsigned char*)bmpData->Scan0.ToPointer();
			int index = 0;
			for (int y = 0; y < imH; y++){
				for (int x = 0; x < imW; x++){
					Sample.X = (double)(x -CenterX) / CenterX;
					Sample.Y= (double)(CenterY-y) / CenterY;
					switch (MethodCodeValue) {
					case 0: //Bayesian-MAP
						ColorScale= 200;
						Output_yi= PClass1*evalPxy1(Sample) -PClass2*evalPxy2(Sample);
						break;
					case 1: //K-NN
						i=(unsigned int) y*imW+x;
						Output_yi= Sgn( (double)(ALLCountClass1[i]-ALLCountClass2[i]) );
						break;
					case 2: //Perceptron Classification
						ColorScale=1;
						Output_yi= PerceptronClassify(Sample);
						break;
					default:
						MessageBox::Show("µL«Øºc¦¹Ãþ§O¤èªk!");
					}//switch
					//Show Contour Type
					LightLevel= 155; //Fixed or Soft „³LightLevel=Min(155, (int)abs(255.0* Output_yi* ColorScale) );
					if (Output_yi>0.0) {
						p[index + 0] = LightLevel; //Red
						p[index + 1] = LightLevel; //Green
						p[index + 2] = 255 -blockcolor; //Blue
					}//if
					else if (Output_yi== 0.0) {
						p[index + 0] = 255; //Red
						p[index + 1] = 255; //Green
						p[index + 2] = 255; //Blue
					}//else if (Output_yi==0)
					else {
						p[index + 0] = 255 -blockcolor; //Red
						p[index + 1] = LightLevel; //Green
						p[index + 2] = LightLevel; //Blue
					}//else
					index += 3;
				}//for x
				index += ByteOfSkip; // ¸õ¹L³Ñ¤UªºPadding bytes
			}//for y
			// Unlock the bits.
			myBitmap->UnlockBits(bmpData);
			pictureBox1->Image = myBitmap;
            pictureBox1->Refresh();
		 }
private: System::Void showMeansToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void showRegressionToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		double LowBound, HighBound, tmpX, tmpY, wk, sumW, s2;
		int X0, Y0;
		int tF = comboBox_P_Function->SelectedIndex;
		LowBound=-1.0; HighBound=1.0;
		X0=0; //tmpX=-1.0 , i.e. shiftX=1.0
		switch (comboBox_regression->SelectedIndex) {
		case 0: //Linear
			tmpY= LR_a0 -LR_a1; //tmpX=-1.0
			break;
		case 1: //Linear--Ln
			tmpY= LR_a0*exp(LR_a1); //tmpX=-1.0 , i.e. shiftX=1.0, ==>Y = a0 * exp^(a1*1.0)
			LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
			break;
		case 2: //Linear¡XLog10
			break;
		case 4: //Nonlinear
			//tmpY=NLcoef[0]-NLcoef[1];
			//for (inti=2; i<NLdegree+1;i++)
			// tmpY += NLcoef[i]*pow(-1.0,i); //tmpX=-1.0
			tmpY=NLcoef[NLdegree];
			for (int i=NLdegree-1; i>=0;i--)
			tmpY= -tmpY+NLcoef[i]; //tmpX=-1.0
			break;
		case 5: //k-NN
			tmpY = 0.0; sumW = 0.0; //tmpX=-1.0
			switch (comboBox_Weight->SelectedIndex){
			case 0: // Y=1/K * Sum_kNNs(InputData[].Y)
				for (int i = 0; i < kNNs; i++)
					tmpY += InputData[NNs[X0][i]].Y;
				tmpY /= kNNs;
				//MessageBox::Show("" + NNs[X0][200]);
				break;
			case 1: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=1/dist_k
				break;
			case 2: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=RBF(dist_k)
				break;
			}//switch(comboBox_Weight->SelectedIndex)
			break;
		case 6:
			tmpY = PerceptronRegression(-1.0, tF);
			break;
		default:
			tmpY= LR_a0-LR_a1; //tmpX=-1.0
		}//switch
		Y0=(int)((HighBound-tmpY)*CenterY);
		while (Y0<1 || Y0 >imH-2 ) {
			X0++;
			switch (comboBox_regression->SelectedIndex) {
			case 0: //Linear
				tmpX= (double) (X0-CenterX)/CenterX;
				tmpY= LR_a0 + LR_a1*tmpX;
				break;
			case 1: //Linear--Ln
				tmpX= (double) (X0+CenterX)/CenterX;
				tmpY= LR_a0*exp(LR_a1*tmpX);
				LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
				break;
			case 4: //Nonlinear
				tmpX = (double) (X0-CenterX)/CenterX; //tmpX=[-1.0,1.0]
				//tmpY=NLcoef[0];
				//for (inti=1; i<NLdegree+1;i++)
				// tmpY+= NLcoef[i]*pow(tmpX,i);
				tmpY=NLcoef[NLdegree];
				for (int i=NLdegree-1; i>=0;i--)
					tmpY= tmpY*tmpX+NLcoef[i];
				break;
			case 5: //k-NN
				break;
			case 6:
				tmpX = (double)(X0 - CenterX) / CenterX;
				tmpY = PerceptronRegression(tmpX, tF);
				break;
			default:
				tmpY= LR_a0 + LR_a1*tmpX;
			}//switch
			Y0=(int)((HighBound-tmpY)*CenterY);
		}//while
		for (int x = X0+1; x < imW-1; x++){
			switch (comboBox_regression->SelectedIndex) {
				case 0: //Linear
					tmpX= (double) (x-CenterX)/CenterX;
					tmpY = LR_a0 + LR_a1 * tmpX; //Y = a0 + a1*X
					break;
				case 1: //Linear--Ln
					tmpX= (double) (x+CenterX)/CenterX;
					tmpY = LR_a0*exp(LR_a1*tmpX); //Y = a0 * exp^(a1*X)
					LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
					break;
				case 4: //Nonlinear
					tmpX= (double) (x-CenterX)/CenterX; //tmpX=[-1.0,1.0]
					//tmpY=NLcoef[0];
					//for (inti=1; i<NLdegree+1;i++)
					//tmpY+= NLcoef[i]*pow(tmpX,i);
					tmpY=NLcoef[NLdegree];
					for (int i=NLdegree-1; i>=0;i--)
						tmpY= tmpY*tmpX+NLcoef[i];
					break;
				case 5: //k-NN
					tmpY = 0.0; sumW = 0.0; //tmpX=-1.0
					switch (comboBox_Weight->SelectedIndex){
						case 0: // Y=1/K * Sum_kNNs(InputData[].Y)
							for (int i = 0; i < kNNs; i++)
								tmpY += InputData[NNs[x][i]].Y;
							tmpY /= kNNs;
							break;
						case 1: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=1/dist_k

							break;
						case 2: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=RBF(dist_k)

							break;
					}//switch(comboBox_Weight->SelectedIndex)
					break;
				case 6:
					tmpX = (double)(X0 - CenterX) / CenterX;
					tmpY = PerceptronRegression(tmpX, tF);
					break;
				default: //Linear
					tmpX= (double) (x-CenterX)/CenterX;
					tmpY = LR_a0 + LR_a1 * tmpX; //Y = a0 + a1*X
			}//switch
			if (tmpY> LowBound&& tmpY< HighBound) {
				X_Cur=x;
				Y_Cur=(int)((HighBound-tmpY)*CenterY); //CenterY=256.0
				penDraw=ClassToPenColor(-1); //Blue
				g->DrawLine(penDraw, X0, Y0, X_Cur, Y_Cur);
				X0=X_Cur;
				Y0=Y_Cur;
			}//if
		}//for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();

		 }
private: System::Void comboBox_regression_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 if (comboBox_regression->SelectedIndex==0 || comboBox_regression->SelectedIndex==4)
				comboBox_NL_degree->Enabled=true; //Nonlinear Regression only use.
			else
				comboBox_NL_degree->Enabled=false; //Nonlinear Regression only use.
		 }
private: System::Void comboBox_Run_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			switch (comboBox_Run->SelectedIndex) {			 
			case 0: //Classification
				comboBox_classify->Enabled=true; //Enable Classification
				comboBox_clustering->Enabled=false; //Disable Clustering
				comboBox_clusters->Enabled=false; //Disable Clustering
				comboBox_regression->Enabled=false; //Disable Regression
				break;
			case 1: //Clustering
				comboBox_clustering->Enabled=true; //Enable Clustering
				comboBox_clusters->Enabled=true;
			case 2: //Regression
				comboBox_regression->Enabled=true; //EnableRegression
				comboBox_NL_degree->Enabled=true;
				break;
			default: //Classification
				comboBox_classify->Enabled=true; //Enable Classification
				comboBox_clustering->Enabled=false; //Disable Clustering
				comboBox_clusters->Enabled=false; //Disable Clustering
				comboBox_regression->Enabled=false; //Disable Regression
			}//switch
		 }
private: System::Void imageToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void groupBox7_Enter(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void comboBox_clustering_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 //Return ClassKindto InputData
			for (int j=0; j<NumberOfData; j++)
				InputData[j].ClassKind=BackupClassKind[j];
			showDataToolStripMenuItem_Click(sender, e);
		 }
private: System::Void showClusteredToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 //Draw Clustered Data
			int r=PointSize/2;
			for (int i=0; i<NumberOfData; i++) {
				X_Cur=(int)(InputData[i].X*CenterX+CenterX); //CenterX=256.0
				Y_Cur=(int)(CenterY-InputData[i].Y*CenterY); //CenterY=256.0
				penDraw=ClassToPenColor(InputData[i].ClassKind);
				switch (InputData[i].ClassKind) {
				case 0: //Circle
					g->DrawEllipse(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
					break;
				case 1: //X
					g->DrawLine(penDraw, X_Cur-r, Y_Cur-r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur+r, Y_Cur-r);
					break;
			    case 2: //Triangle
					g->DrawLine(penDraw, X_Cur, Y_Cur-r, X_Cur-r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur+r, X_Cur, Y_Cur-r);
					break;
				case 3: //Rectangle
					g->DrawRectangle(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
					break;
				case 4: //µÙ§Î
					g->DrawLine(penDraw, X_Cur, Y_Cur-r, X_Cur+r, Y_Cur);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur, X_Cur, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur, Y_Cur+r, X_Cur-r, Y_Cur);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur, X_Cur, Y_Cur-r);
					break;
				case 5: //±è§Î
					g->DrawLine(penDraw, X_Cur-r+1, Y_Cur-r, X_Cur+r-1, Y_Cur-r);
					g->DrawLine(penDraw, X_Cur+r-1, Y_Cur-r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur+r, X_Cur-r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur-r+1, Y_Cur-r);
					break;
			default:
				g->DrawEllipse(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
			}//switch
		}// for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
		 }
private: System::Void showClusterCenterToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
			 int disttmp, j;
			if (checkBox_ShowRange->Checked) {
				//Compute the radius of each cluster
				for (int k=0; k<NumOfClusters; k++)
					Radius[k]= 0;
				for (int i=0; i<NumberOfData; i++) {
					switch (comboBox_clustering->SelectedIndex) {
					case 0: //k-Means
					case 1: //FCM
						disttmp=(int) (sqrt(dist[i][0])*CenterX+0.5);
						if (disttmp> Radius[InputData[i].ClassKind])
							Radius[InputData[i].ClassKind]= disttmp;
						break;
					case 2: //EM
					case 3: //FuzzyGG
						break;
					default: //k-Means
						disttmp=(int) (sqrt(dist[i][0])*CenterX+0.5);
						if (disttmp> Radius[InputData[i].ClassKind])
							Radius[InputData[i].ClassKind]= disttmp;
					}//switch
				} //for i
			}//if
			for (int i=0; i<NumOfClusters; i++) {
				X_Cur=(int)(ClusterCenter[i].X*CenterX+CenterX); //CenterX=256.0
				Y_Cur=(int)(CenterY-ClusterCenter[i].Y*CenterY); //CenterY=256.0
				bshDraw=ClassToColor(ClusterCenter[i].ClassKind);
				g->FillEllipse(bshDraw, X_Cur-PointSize2/2, Y_Cur-PointSize2/2, PointSize2, PointSize2);
				if (checkBox_ShowRange->Checked) {
					penDraw=ClassToPenColor(ClusterCenter[i].ClassKind);
					//(ÃC¦â¡A¶ê¥ª¤W¨¤x®y¼Ð¡A¶ê¥ª¤W¨¤Y®y¼Ð¡AX¶bª½®|¡AY¶bª½®|)
					g->DrawEllipse(penDraw, X_Cur-Radius[i], Y_Cur-Radius[i], 2*Radius[i], 2*Radius[i]);
				}//if
			}//for
			pictureBox1->Image = myBitmap;
			pictureBox1->Refresh();
		 }
private: System::Void comboBox_Kmeans_Option_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void label21_Click(System::Object^  sender, System::EventArgs^  e) {
		 }
private: System::Void comboBox_kNN_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
			 int i;
			 FindMaxKNN();
			kNNs= comboBox_kNN->SelectedIndex*2+1;
			if (comboBox_classify->SelectedIndex==1) {
				if (kNNs>MaxKNN) { //if (kNNs>Number Of Classification Data) set kNNs=3.
				comboBox_kNN->SelectedIndex=0; //if (kNNs>Number Of Classification Data) set kNNs=3.
				kNNs=1;
				MessageBox::Show("k¤Ó¤j¡A¤w¶W¹LMaxKNN³Ì¤j¾F©~¼Æ¶q! ½Ð­«¿ï¡C");
				}//if
			else {
				//Count Neighbors Class Type.
				for (int y = 0; y < imH; y++){
					for (int x = 0; x < imW; x++){
						i= y*imW+x;
						ALLCountClass1[i]=0; ALLCountClass2[i]=0;
						for (int j=0; j<kNNs; j++) {
							if (InputData[ALLNNs[i][j]].ClassKind==1)
								ALLCountClass1[i]++;
							else
								ALLCountClass2[i]++;
						}//for j
					}//for x
				}//for y
			}//else
		}//if(comboBox_classify->SelectedIndex==1)
		else if (kNNs>NumberOfData-1 && comboBox_regression->SelectedIndex==5) {
			comboBox_kNN->SelectedIndex=0; //if (kNNs>Number Of regression Data) set kNNs=1.
			kNNs=1;
			MessageBox::Show("k¤ÓO¤jj¡AA¤ww¶WW¹LLData³ÌI¤jj¾FF©~~¼ÆA¶qq! ½ÐD­«?¿ïi¡CC");
		}//else if

		 }
private: System::Void toolStripButton1_Click(System::Object^  sender, System::EventArgs^  e) {
		 }
};
}

