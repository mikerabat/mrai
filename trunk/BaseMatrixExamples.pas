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

unit BaseMatrixExamples;

// #############################################################
// #### Most Example lists consists of a matrix -> base class for
// #### such learners with direct access to a matrix
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, Matrix;

// #############################################################
// #### most basic element - can be used in an example
type
  TMatrixFeatureList = class(TCustomFeatureList)
  protected
    fOwnsMatrix : boolean;
    fItems : TDoubleMatrix;
    fExampleIdx : integer;

    function GetFeature(index : integer) : double; override;
    procedure SetFeature(index : integer; value : double); override;
  public
    property Data : TDoubleMatrix read fItems;

    procedure SelectCurrent;
    procedure ReleaseCurrentLock;
    procedure SetFeatureVec(const Feature : Array of Double); override;


    constructor Create(aItemList : TDoubleMatrix; ItemIdx : integer; ownsMatrix : Boolean);
    destructor Destroy; override;
  end;

type
  TMatrixLearnerExample = class(TCustomLearnerExample)
  public
    constructor Create(AItemList : TDoubleMatrix; ItemIdx : integer; aClassVal : integer; ownsMatrix : boolean);
  end;

type
  TMatrixExample = class(TCustomExample)
  public
    constructor Create(aMatrix : TDoubleMatrix; ownsMatrix : boolean; ItemIdx : integer = 0);
  end;

type
  TMatrixLearnerExampleList = class(TCustomLearnerExampleList)
  private
    fMatrix: TDoubleMatrix;
    fOwnsMatrix : boolean;
  public
    property Matrix : TDoubleMatrix read fMatrix;
    constructor Create(AItemList : TDoubleMatrix; ClassVals : Array of Integer; ownsMatrix : boolean = True);
    destructor Destroy; override;
  end;

implementation

{ TMatrixFeatureList }

constructor TMatrixFeatureList.Create(aItemList: TDoubleMatrix; ItemIdx: integer; ownsMatrix : Boolean);
begin
     inherited Create;

     fOwnsMatrix := ownsMatrix;
     fItems := aItemList;
     fExampleIdx := ItemIdx;
     fFeatureVecLen := aItemList.Height;
end;

destructor TMatrixFeatureList.Destroy;
begin
     if fOwnsMatrix then
        fItems.Free;
        
     inherited;
end;

function TMatrixFeatureList.GetFeature(index: integer): double;
begin
     Result := fItems[fExampleIdx, index];
end;

procedure TMatrixFeatureList.ReleaseCurrentLock;
begin
     fItems.UseFullMatrix;
end;

procedure TMatrixFeatureList.SelectCurrent;
begin
     fItems.SetSubMatrix(fExampleIdx, 0, 1, fItems.Height);
end;

procedure TMatrixFeatureList.SetFeature(index: integer; value: double);
begin
     fItems[fExampleIdx, index] := value;
end;

procedure TMatrixFeatureList.SetFeatureVec(const Feature: array of Double);
begin
     assert(Length(Feature) = fItems.Height, 'Dimension error');

     fItems.SetRow(fExampleIdx, Feature);
end;

{ TMatrixLearnerExampleList }

constructor TMatrixLearnerExampleList.Create(AItemList: TDoubleMatrix;
  ClassVals: array of Integer; OwnsMatrix : boolean);
var i : integer;
begin
     inherited Create(true);

     assert(Length(ClassVals) = AItemList.Width, 'Dimension Error');
     fOwnsMatrix := ownsMatrix;
     fMatrix := AItemList;

     // ##############################################
     // #### Create item list
     for i := 0 to AItemList.Width - 1 do
         Add(TMatrixLearnerExample.Create(AItemList, i, ClassVals[i], False));
end;

destructor TMatrixLearnerExampleList.Destroy;
begin
     if fOwnsMatrix then
        fMatrix.Free;
        
     inherited;
end;

{ TMatrixLearnerExample }

constructor TMatrixLearnerExample.Create(AItemList: TDoubleMatrix; ItemIdx : integer;
  aClassVal: integer; ownsMatrix : boolean);
begin
     inherited Create(TMatrixFeatureList.Create(AItemList, ItemIdx, ownsMatrix), True);

     ClassVal := aClassVal;
end;

{ TMatrixExample }

constructor TMatrixExample.Create(aMatrix: TDoubleMatrix; ownsMatrix : Boolean; ItemIdx : integer = 0);
begin
     inherited Create(TMatrixFeatureList.Create(aMatrix, ItemIdx, ownsMatrix), True);
end;

end.
