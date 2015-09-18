// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit FischerClassifiers;

// #############################################################
// #### Classifier for the Fischer analysis - both the standard and a robust
// #### implementations
// #############################################################

interface

uses SysUtils, Matrix, Types, BaseClassifier, PCA, BaseMathPersistence;

type
  TOnReconstructExample = procedure(Sender : TObject; Reconstructed : TDoubleMatrix) of Object;

type
  TFischerLDAClassifier = class(TCustomClassifier)
  protected
    fU : TDoubleMatrix;
    fmeanU : TDoubleMatrix;
    fV : TDoubleMatrix;
    fClassCenters : TDoubleMatrixDynArr;
    fClassLabels : TIntegerDynArray;
    fActReaderIndex : integer;
    fOnReconstruct : TOnReconstructExample;
    fIsInList : boolean;

    function ProjectTOPcaSpace(Example : TCustomExample) : TDoubleMatrix; virtual;
    function ProjectToLDASpaceAndClassify(a : TDoubleMatrix) : integer;
  protected
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    procedure OnLoadEndList; override;
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; overload; override;
    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; overload; override;
    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;
    procedure DefineProps; override;
  public
    property OnReconstructFace : TOnReconstructExample read fOnReconstruct write fOnReconstruct;
    function BackProjectedCenters : TDoubleMatrixDynArr; virtual;

    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    constructor Create(U, meanU, V : TDoubleMatrix; const classCenters : TDoubleMatrixDynArr; const classLabels : TIntegerDynArray; doTransPoseU : boolean = True);
    destructor Destroy; override;
  end;

// same as the above classifier but applies a robust PCA recosntruction.
// Ideas taken from - Leonardis, Bischof: Robust Recognition using Eigenimages.
type
  TFischerRobustLDAProps = record
    NumHypothesis : integer;
    Start : Double;              // num Eigenvectors*Start random sampled elements used as initialized pixel set
    Stop : Double;               // num Eigenvectors*Stop elements used as maximum minimal pixel set
    ReductionFactor : double;    // used to iterativly reduce the start*nEig set to stop*nEig
    K2 : double;                 // Used to weight the feature error in the MDL case
    accurFit : boolean;          // Flag which indicates an extended hypothesis fitting process
    maxIter : integer;           // maximum number of refinement steps
    theta : double;              // error threshold for the accurFit procedure
  end;

type
  TFischerRobustLDAClassifier = class(TFischerLDAClassifier)
  private
    fBaseArray : TIntegerDynArray;
    fExample : TCustomExample;

    function RandomSampleFeatures(const NumElem : integer) : TIntegerDynArray;
    function RobustAlphaTrimmedLinEQSolver(var Elements : TIntegerDynArray) : TDoubleMatrix;
    function SimpleMDL(const hypothesis : TDoubleMatrixDynArr) : integer;
    function AccurateFitHypot(a : TDoubleMatrix) : TDoubleMatrix;

    function GenerateHypothesis : TDoubleMatrix;
  protected
    fTheta : double;

    fProps : TFischerRobustLDAProps;
    function ProjectTOPcaSpace(Example : TCustomExample) : TDoubleMatrix; override;
  public
    function BackProjectedCenters : TDoubleMatrixDynArr; override;

    procedure SetProps(const Props : TFischerRobustLDAProps);

    constructor Create(U, meanU, V : TDoubleMatrix; const classCenters : TDoubleMatrixDynArr; const classLabels : TIntegerDynArray;
                       const theta : double);
  end;

// LDA Classifier based on the Fast Robust PCA algorithm
type
  TFischerRobustExLDAClassifier = class(TFischerLDAClassifier)
  private
    fPCA : TFastRobustPCA;
  public
    function BackProjectedCenters : TDoubleMatrixDynArr; override;
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; overload; override;
    procedure DefineProps; override;

    constructor Create(PCA : TFastRobustPCA; V : TDoubleMatrix; const classCenters : TDoubleMatrixDynArr; const classLabels : TIntegerDynArray);
    destructor Destroy; override;
  end;

implementation

uses Classes, FischerBatchLDA, Math, LinearAlgebraicEquations, MatrixConst,
     ClassifierUtils, BaseMatrixExamples, ImageMatrixConv, Graphics, MathUtilFunc;

const cPCAMatrix = 'U';
      cPCAMean = 'meanU';
      cLDAMatrix = 'V';
      cLDAClassCenters = 'classCenters';
      cLDARobustPCA = 'RFisherPCA';
      cLDAClassCenter = 'c';
      cClassLabels = 'labels';
      cPCA = 'PCA';

{ TFischerLDAClassifier }

function TFischerLDAClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var a : TDoubleMatrix;
begin
     assert(Example.FeatureVec.FeatureVecLen = fmeanU.Height, 'Dimension error');
     // ##############################################################
     // #### Project to feature space
     // derrived classes can implement here some kind of robust projection to the feature space
     a := ProjectTOPcaSpace(Example);

     try
        Result := ProjectToLDASpaceAndClassify(a);
     finally
            a.Free;
     end;
end;

constructor TFischerLDAClassifier.Create(U, meanU, V: TDoubleMatrix;
  const classCenters: TDoubleMatrixDynArr;
  const classLabels: TIntegerDynArray; doTransPoseU : boolean);
begin
     assert(High(classLabels) = High(classCenters), 'Dimension error');

     fU := U;
     // we only need the projection to feature space -> transpose immediately
     if doTransPoseU then
        fU.TransposeInPlace;
     fmeanU := meanU;
     fV := V;
     fClassCenters := classCenters;
     fClassLabels := classLabels;

     inherited Create;
end;

procedure TFischerLDAClassifier.DefineProps;
var i : Integer;
begin
     AddObject('FisherU', fU);
     AddObject('FisherMean', fMeanU);
     AddObject('FisherV', fV);

     BeginList(cLDAClassCenters, Length(fClassCenters));
     for i := 0 to Length(fClassCenters) - 1 do
         AddObject(fClassCenters[i]);
     EndList;

     if Length(fClassLabels) > 0 then
        AddIntArr('FisherLabels', fClassLabels);
end;

destructor TFischerLDAClassifier.Destroy;
var i : Integer;
begin
     fU.Free;
     fmeanU.Free;
     fV.Free;

     for i := 0 to Length(fClassCenters) - 1 do
         fClassCenters[i].Free;

     inherited;
end;

procedure TFischerLDAClassifier.OnLoadBeginList(const Name: String;
  count: integer);
begin
     fActReaderIndex := 0;
     fIsInList := False;
     if CompareText(Name, cLDAClassCenters) = 0 then
     begin
          SetLength(fClassCenters, count);
          fIsInList := True;
     end
     else
         inherited;
end;

procedure TFischerLDAClassifier.OnLoadEndList;
begin
     fIsInList := False;
     assert(fActReaderIndex = Length(fClassCenters), 'Error the object could not be read');
end;

procedure TFischerLDAClassifier.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if CompareText(Name, 'FisherLabels') = 0
     then
         fClassLabels := Value
     else
         inherited;
end;

function TFischerLDAClassifier.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := True;

     if CompareText(Name, 'FisherU') = 0
     then
         fU := obj as TDoubleMatrix
     else if CompareText(Name, 'FisherMean') = 0
     then
         fmeanU := obj as TDoubleMatrix
     else if CompareText(Name, 'FisherV') = 0
     then
         fV := obj as TDoubleMatrix
     else 
         Result := inherited OnLoadObject(Name, Obj);
end;

function TFischerLDAClassifier.OnLoadObject(Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if fIsInList
     then
         fClassCenters[fActReaderIndex] := Obj as TDoubleMatrix
     else
         Result := inherited OnLoadObject(Obj);

     inc(fActReaderIndex);
end;

function TFischerLDAClassifier.ProjectToLDASpaceAndClassify(
  a: TDoubleMatrix): integer;
var g : IMatrix;
    distances : TDoubleDynArray;
    i : integer;
    dist : IMatrix;
    minDist : double;
begin
     // ##############################################################
     // #### project to lda space
     g := fV.Mult(a);
     // class is arg min d(g, vi)
     SetLength(distances, Length(fClassCenters));
     for i := 0 to Length(fClassCenters) - 1 do
     begin
          dist := g.Sub(fClassCenters[i]);

          dist.ElementWiseMultInPlace(dist);
          dist.SumInPlace(True);

          distances[i] := dist[0, 0];
     end;

     // ##################################################
     // #### Minimum distance define the class
     Result := fClassLabels[0];
     minDist := distances[0];

     for i := 1 to Length(distances) - 1 do
     begin
          if minDist > distances[i] then
          begin
               minDist := distances[i];
               Result := fClassLabels[i];
          end;
     end;
end;

function TFischerLDAClassifier.ProjectTOPcaSpace(Example: TCustomExample): TDoubleMatrix;
var x0 : IMatrix;
    i : integer;
    rec : TDoubleMatrix;
begin
     // base method: least squares (non robust) prjection
     x0 := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
     for i := 0 to x0.Height - 1 do
         x0[0, i] := Example.FeatureVec[i];
     x0.SubInPlace(fmeanU);
     Result := fU.Mult(x0);

     // for debugging purposes only
     if Assigned(fOnReconstruct) then
     begin
          fU.TransposeInPlace;
          rec := fU.Mult(Result);
          fU.TransposeInPlace;
          rec.AddInplace(fmeanU);

          try
             fOnReconstruct(self, rec);
          finally
                 rec.Free;
          end;
     end;
end;

function TFischerLDAClassifier.BackProjectedCenters: TDoubleMatrixDynArr;
var i : integer;
    vt : IMatrix;
    Ut : IMatrix;
    a : TDoubleMatrix;
begin
     SetLength(Result, Length(fClassCenters));

     vt := fV.Transpose;
     Ut := fU.Transpose;

     for i := 0 to Length(fClassCenters) - 1 do
     begin
          a := vt.Mult(fClassCenters[i]);
          try
             Result[i] := Ut.Mult(a);
             Result[i].AddInPlace(fMeanU);
          finally
                 a.Free;
          end;
     end;
end;

{ TFischerAugmentedLDAClassifier }

function TFischerRobustLDAClassifier.AccurateFitHypot(
  a: TDoubleMatrix): TDoubleMatrix;
var rec : TDoubleMatrix;
    testExmpl : TDoubleMatrix;
    y : Integer;
    numPoints : integer;
    actNumPoints : integer;
    sampleEigVec : TDoubleMatrix;
    i : integer;
    eigVecInv : TDoubleMatrix;
begin
     // create mean free matrix
     eigVecInv := nil;
     rec := nil;
     sampleEigVec := TDoubleMatrix.Create;
     sampleEigVec.Assign(fU);
     testExmpl := TDoubleMatrix.Create(1, fExample.FeatureVec.FeatureVecLen);
     try
        for y := 0 to fExample.FeatureVec.FeatureVecLen - 1 do
            testExmpl[0, y] := fExample.FeatureVec[y] - fmeanU[0, y];

        Result := TDoubleMatrix.Create;
        Result.Assign(a);
        numPoints := testExmpl.Height;

        for i := 0 to fProps.maxIter - 1 do
        begin
             // reconstruct hypothesis
             rec := sampleEigVec.Mult(Result);

             rec.SubInPlace(testExmpl);
             rec.ElementWiseMultInPlace(rec);

             // according to the error construct a new example
             actNumPoints := 0;
             for y := 0 to rec.Height - 1 do
             begin
                  if rec[0, y] < fProps.theta then
                  begin
                       sampleEigVec.SetRow(actNumPoints, sampleEigVec, y);
                       testExmpl[0, actNumPoints] := testExmpl[0, y];
                       inc(actNumPoints);
                  end;
             end;

             if actNumPoints > 0.95*numPoints then
                break;

             numPoints := actNumPoints;
             testExmpl.SetSubMatrix(0, 0, 1, actNumPoints);
             sampleEigVec.SetSubMatrix(0, 0, sampleEigVec.Width, actNumPoints);

             // construct new Result
             if sampleEigVec.PseudoInversion(eigVecInv) <> srOk then
                raise ELDAException.Create('Error finding optimal result');

             Result.Free;
             eigVecInv.MultInPlace(testExmpl);
             Result := eigVecInv;
             eigVecInv := nil;

             FreeAndNil(rec);
        end;
     finally
            rec.Free;
            eigVecInv.Free;
            testExmpl.Free;
            sampleEigVec.Free;
     end;
end;

function TFischerRobustLDAClassifier.BackProjectedCenters: TDoubleMatrixDynArr;
var i : integer;
    vt : TDoubleMatrix;
    a : TDoubleMatrix;
begin
     // note: this time the matrix U is already in it's original form
     SetLength(Result, Length(fClassCenters));

     vt := fV.Transpose;
     try
        for i := 0 to Length(fClassCenters) - 1 do
        begin
             a := vt.Mult(fClassCenters[i]);
             try
                Result[i] := fU.Mult(a);
                Result[i].AddInPlace(fMeanU);
             finally
                    a.Free;
             end;
        end;
     finally
            vt.Free;
     end;
end;

constructor TFischerRobustLDAClassifier.Create(U, meanU, V: TDoubleMatrix;
  const classCenters: TDoubleMatrixDynArr; const classLabels: TIntegerDynArray;
  const theta: double);
begin
     inherited Create(U, meanU, V, classCenters, classLabels, False);

     fTheta := theta;
     fProps.NumHypothesis := 1;
     fProps.ReductionFactor := 0.85;
     fProps.K2 := 0.01;
     fProps.Start := 50;
     fProps.Stop := 20;
     fProps.accurFit := False;
     fProps.theta := 100;
     fProps.maxIter := 3;
end;

function TFischerRobustLDAClassifier.GenerateHypothesis: TDoubleMatrix;
var elements : TIntegerDynArray;
begin
     elements := RandomSampleFeatures(Min(fExample.FeatureVec.FeatureVecLen, Trunc(fU.Width*fProps.Start)));
     Result := RobustAlphaTrimmedLinEQSolver(elements);
end;

function TFischerRobustLDAClassifier.ProjectTOPcaSpace(
  Example: TCustomExample): TDoubleMatrix;
var i : integer;
    hypothesis : TDoubleMatrixDynArr;
    hypIdx : integer;
    rec : TDoubleMatrix;
begin
     fProps.NumHypothesis := Max(1, fProps.NumHypothesis);
     fExample := Example;

     // #################################################################
     // #### Classification is a bit different here
     SetLength(fBaseArray, Example.FeatureVec.FeatureVecLen);
     for i := 0 to Example.FeatureVec.FeatureVecLen - 1 do
         fBaseArray[i] := i;

     // #################################################################
     // #### Generate hypothesis
     SetLength(hypothesis, fProps.NumHypothesis);
     try
        for i := 0 to fProps.NumHypothesis - 1 do
            hypothesis[i] := GenerateHypothesis;

        // #################################################################
        // #### Minimum Description length criteria to find out which is the best suited one
        if fProps.NumHypothesis > 1
        then
            hypIdx := SimpleMDL(hypothesis)
        else
            hypIdx := 0;

        Result := hypothesis[hypIdx];
        hypothesis[hypIdx] := nil;
     finally
            for i := 0 to Length(hypothesis) - 1 do
                hypothesis[i].Free;
     end;

     if fProps.accurFit then
     begin
          rec := AccurateFitHypot(Result);
          Result.Free;
          Result := rec;
     end;

     // for debugging purposes only
     if Assigned(fOnReconstruct) then
     begin
          rec := fU.Mult(Result);
          rec.AddInplace(fmeanU);

          try
             fOnReconstruct(self, rec);
          finally
                 rec.Free;
          end;
     end;
end;


function TFischerRobustLDAClassifier.RandomSampleFeatures(
  const NumElem: integer): TIntegerDynArray;
var i : integer;
    len : integer;
begin
     // ################################################################
     // #### Ensure no double indices
     //len := Length(fBaseArray);
//     Result := Copy(fBaseArray, 0, len);
//
//     // check if we realy have to randomly sample the values
//     numRetValues := Trunc(fU.Width*fProps.Start);
//     if Length(Result) > numRetValues then
//     begin
//          for i := 0 to len*2 - 1 do
//          begin
//               idx1 := Random(len);
//               idx2 := Random(len);
//
//               help := Result[idx1];
//               Result[idx1] := Result[idx2];
//               Result[idx2] := help;
//          end;
//     end;
// SetLength(Result, Min(Length(Result), numRetValues));

// the paper itself allows double indices -> thus we use the faster method here
     SetLength(Result, NumElem);

     len := fU.Height;
     for i := 0 to NumElem - 1 do
         Result[i] := random(len);
end;

function TFischerRobustLDAClassifier.RobustAlphaTrimmedLinEQSolver(
  var Elements: TIntegerDynArray): TDoubleMatrix;
var numElements : integer;
    A : TDoubleMatrix;
    AInv : TDoubleMatrix;
    Y : TDoubleMatrix;
    X : TDoubleMatrix;
    Xa : TDoubleMatrix;
    i : integer;
    data : TDoubleDynArray;
    thresh : double;
    tIdx : integer;
    dataIdx : integer;
    meanNormExample : TDoubleMatrix;
    numNewElements : integer;
begin
     Result := nil;

     numElements := Length(Elements);

     A := nil;
     Y := nil;
     x := nil;
     Xa := nil;
     AInv := nil;
     meanNormExample := TDoubleMatrix.Create(1, fExample.FeatureVec.FeatureVecLen);
     try
        for i := 0 to meanNormExample.Height - 1 do
            meanNormExample[0, i] := fExample.FeatureVec[i] - fmeanU[0, i];

        repeat
              A := TDoubleMatrix.Create(fU.Width, numElements);
              X := TDoubleMatrix.Create(1, numElements);

              // copy rows respectively elements
              for i := 0 to numElements - 1 do
              begin
                   A.SetRow(i, fU, Elements[i]);
                   X[0, i] := meanNormExample[0, Elements[i]];
              end;

              // ##########################################################
              // #### solve this overdetermined linear set of equations

              // todo: eventually dismiss the pseudoinverse and apply a better
              // solving method
              if A.PseudoInversion(AInv) <> srOk then
                 raise ELDAException.Create('Error robustly solving LDA coefficients');
              AInv.MultInPlace(X);

              // ##########################################################
              // #### Check error distribution -> but only on the projected subset
              Xa := A.Mult(AInv);

              // check error
              for i := 0 to numElements - 1 do
                  X[0, i] := Abs(X[0, i] - Xa[0, i]);

              // reduce the number of points according to the error distribution
              data := X.SubMatrix;
              QuickSort(data[0], sizeof(double), Length(data), DoubleSortFunc);

              numNewElements := Max(fU.Width, Min(numElements - 1, Round(numElements*fProps.ReductionFactor)));
              if numNewElements <= 0 then
                 break;

              thresh := data[numNewElements];
              dataIdx := 0;

              for tidx := 0 to numElements - 1 do
              begin
                   if dataIdx = numNewElements - 1 then
                      break;
                      
                   if (X[0, tidx] <= thresh) then
                   begin
                        Elements[dataIdx] := Elements[tidx];
                        inc(dataIdx);
                   end;
              end;

              numElements := dataIdx;

              // ##########################################################
              // #### Create result
              if numElements <= Max(fU.Width, fProps.Stop*fU.Width) then
              begin
                   Result := AInv;
                   Ainv := nil;
              end;

              FreeAndNil(Xa);
              FreeAndNil(X);
              FreeAndNil(A);
              FreeAndNil(aInv);
        until numElements <= Max(fU.Width, fProps.Stop*fU.Width);

        FreeAndNil(meanNormExample);
     except
           A.Free;
           Y.Free;
           X.Free;
           Xa.Free;
           AInv.Free;
           meanNormExample.Free;

           raise;
     end;
end;

procedure TFischerRobustLDAClassifier.SetProps(
  const Props: TFischerRobustLDAProps);
begin
     fProps := Props;
end;


function TFischerRobustLDAClassifier.SimpleMDL(const hypothesis: TDoubleMatrixDynArr): integer;
var i, y : integer;
    exmplMeanReduced : TDoubleMatrix;
    maxMDL : double;
    numFeatures : integer;
    error : double;
    backProj : TDoubleMatrix;
    mdl : double;
begin
     Result := 0;

     // ############################################################
     // #### compute cost function of each hypothesis
     exmplMeanReduced := TDoubleMatrix.Create(1, fExample.FeatureVec.FeatureVecLen);
     try
        for y := 0 to fExample.FeatureVec.FeatureVecLen - 1 do
            exmplMeanReduced[0, y] := fExample.FeatureVec[y] - fmeanU[0, y];

        maxMDL := -MaxDouble;
        for i := 0 to Length(hypothesis) - 1 do
        begin
             // check error and number of points smaller than theta
             error := 0;
             numFeatures := 0;

             // backproject hypothesis
             backProj := fU.Mult(hypothesis[i]);
             try
                for y := 0 to exmplMeanReduced.Height - 1 do
                begin
                     error := error + sqr(exmplMeanReduced[0, y] - backProj[0, y]);

                     // todo: in the original code theta is a different one
                     // but I think it's quite similar to the created one
                     // normally the error is much higher than the number of points
                     // thus the criteria will give back the hypothesis yielding the lowest error
                     // most of the time.
                     if abs(exmplMeanReduced[0, y] - backProj[0, y]) < fTheta then
                        inc(numFeatures);
                end;

                mdl := numFeatures - fProps.K2*error;

                if mdl > maxMdl then
                begin
                     Result := i;
                     maxMdl := mdl;
                end;
             finally
                    backProj.Free;
             end;
        end;
     finally
            exmplMeanReduced.Free;
     end;
end;

{ TFischerRobustExLDAClassifier }

function TFischerRobustExLDAClassifier.BackProjectedCenters: TDoubleMatrixDynArr;
var i : integer;
    vt : TDoubleMatrix;
    a : TDoubleMatrix;
begin
     // note: this time the matrix U is already in it's original form
     SetLength(Result, Length(fClassCenters));

     vt := fV.Transpose;
     try
        for i := 0 to Length(fClassCenters) - 1 do
        begin
             a := vt.Mult(fClassCenters[i]);
             try
                Result[i] := fPCA.Reconstruct(a);
             finally
                    a.Free;
             end;
        end;
     finally
            vt.Free;
     end;
end;


function TFischerRobustExLDAClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var exmpl : TDoubleMatrix;
    a : TDoubleMatrix;
    i : integer;
    rec : TDoubleMatrix;
begin
     // #################################################################
     // #### robustly project the example to the feature space
     if (Example.FeatureVec is TMatrixFeatureList) then
     begin
          TMatrixFeatureList(Example.FeatureVec).SelectCurrent;
          a := fPCA.ProjectToFeatureSpace(TMatrixFeatureList(Example.FeatureVec).Data);
          TMatrixFeatureList(Example.FeatureVec).ReleaseCurrentLock;
     end
     else
     begin
          exmpl := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
          try
             for i := 0 to Example.FeatureVec.FeatureVecLen - 1 do
                 exmpl[0, i] := Example.FeatureVec[i];

             a := fPCA.ProjectToFeatureSpace(exmpl);
          finally
                 exmpl.Free;
          end;
     end;

     // for debugging purposes only
     if Assigned(fOnReconstruct) then
     begin
          rec := fPCA.Reconstruct(a);
          try
             fOnReconstruct(self, rec);
          finally
                 rec.Free;
          end;
     end;

     // #################################################################
     // #### now we can perform a "normal" lda classification
     try
        Result := ProjectToLDASpaceAndClassify(a);
     finally
            a.Free;
     end;
end;

constructor TFischerRobustExLDAClassifier.Create(PCA: TFastRobustPCA;
  V: TDoubleMatrix; const classCenters: TDoubleMatrixDynArr;
  const classLabels: TIntegerDynArray);
begin
     fPCA := PCA;
     fV := V;
     fClassCenters := classCenters;
     fClassLabels := classLabels;     
end;

procedure TFischerRobustExLDAClassifier.DefineProps;
begin
     inherited;

     AddObject(cLDARobustPCA, fPCA);
end;

destructor TFischerRobustExLDAClassifier.Destroy;
begin
     fPCA.Free;

     inherited;
end;

function TFischerRobustExLDAClassifier.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := True;

     if CompareText(Name, cLDARobustPCA) = 0
     then
         fPCA := obj as TFastRobustPCA
     else
         Result := inherited OnLoadObject(Name, Obj);
end;

initialization
  RegisterMathIO(TFischerLDAClassifier);
  RegisterMathIO(TFischerRobustLDAClassifier);
  RegisterMathIO(TFischerRobustExLDAClassifier);


end.
