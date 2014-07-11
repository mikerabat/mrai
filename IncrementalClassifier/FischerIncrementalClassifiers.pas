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

unit FischerIncrementalClassifiers;

// #############################################################
// #### Classifier for the incremental Fischer learning algorithm
// ####- both the standard and a robust implementations
// #############################################################

interface

uses SysUtils, Matrix, Types, BaseClassifier, IncrementalPCA, PCA, FischerClassifiers, BaseMathPersistence;

// #############################################################
// #### Incremental classifier data holder + classification procedure
type
  TIncrementalFischerLDAClassifier = class(TCustomClassifier)
  private
    fPCA : TMatrixPCA;

    fV : TDoubleMatrix;
    fClassCenters : TDoubleMatrixDynArr;
    fClassLabels : TIntegerDynArray;
    fNumLabels : integer;
    fTheta : double;
    fClassIdx: TIntegerDynArray;
    fActReaderIndex : integer;
    fOnReconstruct: TOnReconstructExample;
    procedure SetClassLabels(const Value: TIntegerDynArray);
    procedure SetClassIdx(const Value: TIntegerDynArray);
    function ProjectToLDASpaceAndClassify(a: TDoubleMatrix): integer;
    function GetPCA : TMatrixPCA;
  protected
    function CreatePCA: TMatrixPCA; virtual;
  public
    property PCAData : TMatrixPCA read GetPCA;
    property ClassIdx : TIntegerDynArray read fClassIdx write SetClassIdx;
    property V : TDoubleMatrix read fV;
    property Theta : double read fTheta write fTheta;
    property Classcenters : TDoubleMatrixDynArr read fClassCenters write fClassCenters;
    property ClassLabels : TIntegerDynArray read fClassLabels write SetClassLabels;
    property NumLabels : integer read fNumLabels;

    property OnReconstructFace : TOnReconstructExample read fOnReconstruct write fOnReconstruct;
    function BackProjectedCenters : TDoubleMatrixDynArr;

    procedure AddClassIdx(alabel : integer);

    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; overload; override;
    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; overload; override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    procedure OnLoadEndList; override;

    // ###########################################################
    // #### abstract functions -> loading of data + classification
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;
    procedure DefineProps; override;

    constructor Create;
    destructor Destroy; override;
  end;

// ##########################################################
// #### Same as base class but with a robust pca data holder
type
  TFastRobustIncrementalLDAClassifier = class(TIncrementalFischerLDAClassifier)
  protected
    function CreatePCA : TMatrixPCA; override;
  end;


implementation

uses Classes, BaseMatrixExamples;

const cPCA = 'PCA';
      cLDAMatrix = 'V';
      cLDAClassCenters = 'classCenters';
      cLDAClassCenter = 'c';
      cClassLabels = 'labels';
      cLDAExmplIdx = 'ClassIDX';
      cLDAExmpl = 'IDX';
      cPCANumVecs = 'PCANumKeepVecs';


procedure TIncrementalFischerLDAClassifier.OnLoadBeginList(
  const Name: String; count: integer);
begin
     fActReaderIndex := -1;
     if CompareText(Name, cLDAClassCenters) = 0 then
     begin
          SetLength(fClassCenters, count);
          fActReaderIndex := 0;
     end
     else
         inherited;
end;

procedure TIncrementalFischerLDAClassifier.OnLoadEndList;
begin
     inherited;
end;

procedure TIncrementalFischerLDAClassifier.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if CompareText(Name, cLDAExmpl) = 0
     then
         fClassIdx := Value
     else if CompareText(Name, cClassLabels) = 0
     then
         fClassLabels := Value
     else
         inherited;
end;

function TIncrementalFischerLDAClassifier.OnLoadObject(
  Obj: TBaseMathPersistence): boolean;
begin
     if fActReaderIndex >= 0 then
     begin
          fClassCenters[fActReaderIndex] := obj as TDoubleMatrix;
          inc(fActReaderIndex);
          Result := True;
     end
     else
         Result := inherited OnLoadObject(obj);
end;

function TIncrementalFischerLDAClassifier.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if CompareText(Name, cPCA) = 0
     then
         fPCA := obj as TMatrixPCA
     else if CompareText(Name, cLDAMatrix) = 0
     then
         fV := obj as TDoubleMatrix
     else
         Result := Inherited OnLoadObject(Name, Obj);
end;

{ TIncrementalFischerLDAClassifier }
procedure TIncrementalFischerLDAClassifier.DefineProps;
var i : Integer;
begin
     AddObject(cPCA, fPCA);
     AddObject(cLDAMatrix, fV);

     BeginList(cLDAClassCenters, Length(fClassCenters));
     for i := 0 to Length(fClassCenters) - 1 do
         AddObject(fClassCenters[i]);
     EndList;

     AddIntArr(cClassLabels, fClassLabels);
     if Length(fClassIdx) > 0 then
        AddIntArr(cLDAExmpl, fClassIdx);
end;

procedure TIncrementalFischerLDAClassifier.AddClassIDx(alabel: integer);
begin
     if Length(fClassIdx) <= fNumLabels + 1 then
        SetLength(fClassIdx, 100 + Length(fClassIdx));

     fClassIdx[fNumLabels] := aLabel;
     inc(fNumLabels);
end;

function TIncrementalFischerLDAClassifier.BackProjectedCenters: TDoubleMatrixDynArr;
var i : integer;
    a : IMatrix;
    vt : IMatrix;
begin
     SetLength(Result, Length(fClassCenters));

     vt := fV.Transpose;

     for i := 0 to Length(Result) - 1 do
     begin
          a := vt.Mult(fClassCenters[i]);
          Result[i] := fPCA.Reconstruct(a.GetObjRef);
     end;
end;

function TIncrementalFischerLDAClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var a : TDoubleMatrix;
    x0 : TDoubleMatrix;
    y : Integer;
    ownsMatrix : boolean;
    recon : TDoubleMatrix;
begin
     assert(Example.FeatureVec.FeatureVecLen = PCAData.Mean.Height, 'Dimension error');
     // ##############################################################
     // #### Project to feature space
     // derrived classes can implement here some kind of robust projection to the feature space
     ownsMatrix := False;
     if Example is TMatrixExample
     then
         x0 := (TMatrixExample(Example).FeatureVec as TMatrixFeatureList).Data
     else
     begin
          ownsMatrix := True;
          x0 := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
          for y := 0 to x0.Height - 1 do
              x0[0, y] := Example.FeatureVec[y];
     end;
     try
        a := PCAData.ProjectToFeatureSpace(x0);
        try
           Result := ProjectToLDASpaceAndClassify(a);

           // for debugging purposes only
           if Assigned(fOnReconstruct) then
           begin
                recon := PCAData.Reconstruct(a);

                try
                   fOnReconstruct(Self, recon);
                finally
                       recon.Free;
                end;
           end;
           
        finally
               a.Free;
        end;
     finally
            if ownsMatrix then
               x0.Free;
     end;
end;

function TIncrementalFischerLDAClassifier.ProjectToLDASpaceAndClassify(
  a: TDoubleMatrix): integer;
var g : TDoubleMatrix;
    distances : TDoubleDynArray;
    i : integer;
    dist : TDoubleMatrix;
    minDist : double;
begin
     // ##############################################################
     // #### project to lda space
     g := fV.Mult(a);
     try
        // class is arg min d(g, vi)
        SetLength(distances, Length(fClassCenters));
        for i := 0 to Length(fClassCenters) - 1 do
        begin
             dist := g.Sub(fClassCenters[i]);
             try
                dist.ElementWiseMultInPlace(dist);
                dist.SumInPlace(True);

                distances[i] := dist[0, 0];
             finally
                    dist.Free;
             end;
        end;
     finally
            g.Free;
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

constructor TIncrementalFischerLDAClassifier.Create;
begin
     fPCA := nil;

     fV := TDoubleMatrix.Create;
     fClassCenters := nil;
     fClassLabels := nil;
end;

function TIncrementalFischerLDAClassifier.CreatePCA: TMatrixPCA;
begin
     Result := TIncrementalPCA.Create;
end;

destructor TIncrementalFischerLDAClassifier.Destroy;
var  i: Integer;
begin
     fPCA.Free;
     fV.Free;

     for i := 0 to Length(fClassCenters) - 1 do
         fClassCenters[i].Free;
end;

function TIncrementalFischerLDAClassifier.GetPCA: TMatrixPCA;
begin
     if not Assigned(fPCA) then
        fPCA := CreatePCA;

     Result := fPCA;
end;

procedure TIncrementalFischerLDAClassifier.SetClassIdx(
  const Value: TIntegerDynArray);
begin
     fClassIdx := Value;
     fNumLabels := Length(fClassIdx);
end;

procedure TIncrementalFischerLDAClassifier.SetClassLabels(
  const Value: TIntegerDynArray);
begin
     fClassLabels := Value;
end;

{ TFastRobustIncrementalLDAClassifier }

function TFastRobustIncrementalLDAClassifier.CreatePCA: TMatrixPCA;
begin
     Result := TFastRobustIncrementalPCA.Create;
end;

end.
