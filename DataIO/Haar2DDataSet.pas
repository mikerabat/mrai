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

unit Haar2DDataSet;

// ##############################################################
// #### Haar like features data sets
// ##############################################################

interface

uses Windows, SysUtils, Contnrs, BaseClassifier, IntegralImg, Haar2D,
     Matrix, RandomEng;

type
  THaar2DLearnerList = class(TCustomLearnerExampleList)
  private
    fIntImages : TObjectList;
  public
    procedure AddIntImg(img : TIntegralImage);
    procedure AddExample(list : THaar2DFeatureList; classVal : integer);

    function CloneBase : TCustomLearnerExampleList; override;

    constructor Create;
    destructor Destroy; override;
  end;

// #############################################################
// #### Example list creation for the haar like features boosting learners
type
  THaar2DLearnerExampleListCreator = class(TObject)
  private
    const cMinNumNegExamples : integer = 40;
  private
    fWinWidth : integer;
    fWinHeight : integer;
    fLearnerList : THaar2DLearnerList;
    fIntImg : TIntegralImage;                    // holds a refernce to the last added integral image
    fRndEng : TRandomGenerator;

    fRefFeatureList : THaar2DRefFeatureList;
    procedure Haar2DImgLoad(Sender : TObject; mtx : TDoubleMatrix; actNum, NumImags : integer; const FileName : string);
  public
    property LearnerList : THaar2DLearnerList read fLearnerList;

    procedure InitNewImg(img : IMatrix);
    procedure AddExample(rc : TRect; classVal : integer);
    function Finish : TCustomLearnerExampleList;

    procedure InitFromDir(const dir : string; doRecursive : Boolean; allFilesSameSize : boolean = False);

    constructor Create(winWidth, winHeight : integer; numColPlane : integer; featureList : TIntegralType);
    destructor Destroy; override;
  end;

implementation

uses MatrixImageLists, ImageMatrixConv, math, BaseIncrementalLearner,
     BaseMatrixExamples, Classes;

{ THaar2DLearnerExampleListCreator }

procedure THaar2DLearnerExampleListCreator.InitFromDir(const dir: string; doRecursive : Boolean; allFilesSameSize : boolean = False);
var colType : TimageConvType;
begin
     // initialize the list from a directory containing images + the corresponding
     // object coordinates

     // the object description file is a text file containing a comma separated list
     // of coordinates. The file itself may contain more than one example!
     if fRefFeatureList.NumColorPlanes = 3
     then
         colType := ctRGB
     else
         colType := ctGrayScale;

     with TIncrementalImageList.Create do
     try
        Recursive := doRecursive;
        OnImageStep := Haar2DImgLoad;

        if allFilesSameSize 
        then
            ReadListFromDirectory(dir, colType)
        else
            ReadListFromDirectoryRaw(dir, colType);
            
     finally
            Free;
     end;
end;

procedure THaar2DLearnerExampleListCreator.InitNewImg(img: IMatrix);
var intImg : TIntegralImage;
begin
     intImg := TIntegralImage.Create(img, fRefFeatureList.FeatureTypes, fRefFeatureList.NumColorPlanes);

     if not Assigned(fLearnerList) then
        fLearnerList := THaar2DLearnerList.Create;

     fLearnerList.AddIntImg(intImg);
     fIntImg := intImg;
end;

procedure THaar2DLearnerExampleListCreator.AddExample(rc: TRect;
  classVal: integer);
var featureList : THaar2DFeatureList;
begin
     assert(Assigned(fLearnerList), 'Call InitNewImg first!');
     assert(rc.Right - rc.Left > 3, 'Error the given rectangle is too small');
     assert(rc.Bottom - rc.Top > 3, 'Error the given rectangle is too small');
     featureList := THaar2DFeatureList.Create(fIntImg, fRefFeatureList, fRefFeatureList.NumColorPlanes);

     featureList.WindowX := rc.Left;
     featureList.windowY := rc.Top;
     featureList.Scale := (rc.Right - rc.Left)/fRefFeatureList.WindowWidth;

     fLearnerList.AddExample(featureList, classVal);
end;

constructor THaar2DLearnerExampleListCreator.Create(winWidth, winHeight : integer; numColPlane : integer; featureList : TIntegralType);
begin
     inherited Create;

     fRndEng := TRandomGenerator.Create;
     fRndEng.RandMethod := raMersenneTwister;
     fRndEng.Init(0);

     fWinWidth := winWidth;
     fWinHeight := winHeight;

     fLearnerList := nil;
     fRefFeatureList := THaar2DRefFeatureList.Create(winWidth, winHeight, numColPlane, featureList);
end;

destructor THaar2DLearnerExampleListCreator.Destroy;
begin
     fLearnerList.Free;
     fRefFeatureList.Free;
     fRndEng.Free;

     inherited;
end;

function THaar2DLearnerExampleListCreator.Finish: TCustomLearnerExampleList;
begin
     Result := fLearnerList;

     // the callee owns this list now!
     fLearnerList := nil;
end;

procedure THaar2DLearnerExampleListCreator.Haar2DImgLoad(Sender : TObject; mtx : TDoubleMatrix; actNum, NumImags : integer;
  const FileName : string);
var clfile : string;
    sl1, sl2 : TStringList;
    counter : Integer;
    m : IMatrix;
    rcList : Array of TRect;
    rc : TRect;
    inserted : boolean;
    w, h : integer;
    numIter : integer;
function OverlapRec(const rc : TRect) : boolean;
var i : integer;
    uRc : TRect;
    isOverlap : boolean;
begin
     Result := False;

     for i := 0 to Length(rcList) - 1 do
     begin
          isOverlap := IntersectRect(uRc, rcList[i], rc);

          isOverlap := isOverlap and (w*h < (uRc.Right - uRc.Left)*(uRc.Bottom - uRc.Top)*30);
          Result := Result or isOverlap;

          if Result then
             break;
     end;
end;
begin
     clFile := ChangeFileExt(FileName, '.pos');

     // create a copy so we "own" the data
     m := mtx.Clone;
     InitNewImg(m);

     rcList := nil;

     // load the associated example list
     if FileExists(FileName) then
     begin
          // file format is a comma separated list of lines. Each line consists of
          // 4 values in form (x1, y1, x2, y2) - these four coordinates define the
          // object borders
          try
             sl1 := TStringList.Create;
             sl2 := TStringList.Create;
             try
                sl1.LoadFromFile(clFile);

                SetLength(rcList, sl1.Count);
                for counter := 0 to sl1.Count - 1 do
                begin
                     sl2.CommaText := sl1[counter];

                     assert(sl2.Count >= 4, 'Error number of elements do not match');
                     rcList[counter] := Rect(StrToInt(sl2[0]), StrToInt(sl2[1]), StrToInt(sl2[2]), StrToInt(sl2[3]));

                     AddExample(rcList[counter], 1);
                end;
             finally
                    sl1.Free;
                    sl2.Free;
             end;
          except
                on E : Exception do
                begin
                     raise Exception.Create('Error in File "' + clFile + '": ' + E.Message);
                end;
          end;
     end;

     // check if there is a list of "negative" examples
     // use them instead of a random list of rectangles
     clFile := ChangeFileExt(FileName, '.neg');
     if FileExists(clFile) then
     begin
          // file format is the same as the positive example list
          try
             sl1 := TStringList.Create;
             sl2 := TStringList.Create;
             try
                sl1.LoadFromFile(clFile);

                SetLength(rcList, sl1.Count);
                for counter := 0 to sl1.Count - 1 do
                begin
                     sl2.CommaText := sl1[counter];

                     assert(sl2.Count >= 4, 'Error number of elements do not match');
                     rcList[counter] := Rect(StrToInt(sl2[0]), StrToInt(sl2[1]), StrToInt(sl2[2]), StrToInt(sl2[3]));
                     AddExample(rcList[counter], -1);
                end;
             finally
                    sl1.Free;
                    sl2.Free;
             end;
          except
                on E : Exception do
                begin
                     raise Exception.Create('Error in File "' + clFile + '": ' + E.Message);
                end;
          end;
     end
     else
     begin
          w := rcList[0].Right - rcList[0].Left;
          h := rcList[0].Bottom - rcList[0].Top;

          sl1 := TStringList.Create;


          (*
          for counter := 0 to 500 - 1 do
          begin
               rc.Left := counter mod 50;
               rc.Top := counter div 50;

               rc.Right := rc.Left + w - 1;
               rc.Bottom := rc.Top + h - 1;

               sl1.Add(IntToStr(rc.Left) + ',' + IntToStr(rc.Top) + ',' + IntToStr(rc.Right) + ',' + IntToStr(rc.Bottom));
               AddExample(rc, -1);
          end;
         sl1.Free
          *)


          // add negative examples - use random examples and check if they do not overlap too much with the object (max 10%)
          // write out the negative example list!
          try
             for counter := 0 to Max(cMinNumNegExamples, 2*Length(rcList)) - 1 do
             begin
                  numIter := 0;
                  repeat
                        rc.Left := fRndEng.RandInt(m.Width - w - 1);
                        rc.Top := fRndEng.RandInt(m.Height - h - 1);
                        inc(numIter);

                        rc.Right := rc.left + w - 1;
                        rc.Bottom := rc.Top + h - 1;

                        inserted := not OverlapRec(rc);
                  until (numIter = 5) or inserted;

                  if inserted then
                  begin
                       sl1.Add(IntToStr(rc.Left) + ',' + IntToStr(rc.Top) + ',' + IntToStr(rc.Right) + ',' + IntToStr(rc.Bottom));
                       AddExample(rc, -1);
                  end;
             end;

             sl1.SaveToFile(clfile);
          finally
                 sl1.Free;
          end;  // *)
     end;
end;

{ THaar2DLearnerList }

procedure THaar2DLearnerList.AddExample(list: THaar2DFeatureList;
  classVal: integer);
begin
     Add(THaar2DLearnerExample.Create(list, classVal));
end;

procedure THaar2DLearnerList.AddIntImg(img: TIntegralImage);
begin
     // the learner list owns the objects - the integral images get's shared
     // between the examples
     fIntImages.Add(img);
end;

constructor THaar2DLearnerList.Create;
begin
     fIntImages := TObjectList.Create(True);

     inherited Create;
end;

destructor THaar2DLearnerList.Destroy;
begin
     fIntImages.Free;

     inherited;
end;

function THaar2DLearnerList.CloneBase: TCustomLearnerExampleList;
begin
     Result := inherited CloneBase;
     THaar2DLearnerList(Result).fIntImages.OwnsObjects := False;
     THaar2DLearnerList(Result).fIntImages.Assign(fIntImages);  // just copy references
end;

end.
