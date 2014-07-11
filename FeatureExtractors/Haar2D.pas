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

unit Haar2D;

// ###########################################################
// #### 2d Haar features
// ###########################################################

interface

uses SysUtils, IntegralImg, BaseClassifier, Matrix, contnrs;

// ###########################################################
// #### 2D Haar like feature list
type
  THaar2DFeature = class(TObject)
  private
    const cMaxFeatureRects = 3;

    type
      THaarRect = record
        x, y, width, height : integer;
        weight : integer;
      end;
  private
    fRects : Array[0..cMaxFeatureRects-1] of THaarRect;
    fIsTilted : boolean;
  public
    property Tilted : boolean read fIsTilted;

    function EvalFeature(img : TIntegralImage; x, y : integer; scale : double; colPlane : integer) : double;

    constructor Create(tilted: boolean; x0, y0, w0, h0, wt0, x1, y1, w1, h1, wt1 : integer); overload;
    constructor Create(tilted: boolean; x0, y0, w0, h0, wt0, x1, y1, w1, h1, wt1, x2, y2, w2, h2, wt2 : integer); overload;
  end;

// ############################################################
// #### Since the base types are always the same create a reference list
// -> all feature lists can reference to this one
type
  THaar2DRefFeatureList = class(TObjectList)
  private
    fWindowWidth, fWindowHeight : integer;
    fNumColorPlanes : integer;
    fFeatureTypes : TIntegralType;
  public
    property NumColorPlanes : integer read fNumColorPlanes;
    property WindowWidth : integer read fWindowWidth;
    property WindowHeight : integer read fWindowHeight;
    property FeatureTypes : TIntegralType read fFeatureTypes;

    function Feature(idx : integer) : THaar2DFeature; inline;

    constructor Create(initWidth, initHeight : integer; numColorPlanes : integer; aFeatureTypes : TIntegralType);
  end;

// ###############################################################
// #### feature list implementation - create a framework compatible version
// #### of the haar 2d feature list
type
  THaar2DFeatureList = class(TCustomFeatureList)
  private
    const cSigmaFact = 2;   // from lienhardt et. all
  private
    fIntImg : TIntegralImage;
    fRefList : THaar2DRefFeatureList;
    fX, fY : integer;
    fScale : double;
    fNumColorPlanes : integer;
    fNormFact : double;

    procedure SetX(const Value: integer);
    procedure SetY(const Value: integer);
    procedure SetScale(const Value: double);
  protected
    function GetFeature(index : integer) : double; override;
    procedure SetFeature(index : integer; value : double); override;
  public
    property WindowX : integer read fX write SetX;
    property WindowY : integer read fY write SetY;
    property Scale : double read fScale write SetScale;

    procedure SetFeatureVec(const Feature : Array of Double); override;

    constructor Create(img : TIntegralImage; refFeatureList : THaar2DRefFeatureList; numColorPlanes : integer);
    destructor Destroy; override;
  end;

// ##############################################################
// #### learning object definitions
type
  THaar2DExample = class(TCustomExample)
  public
    constructor Create(featureList : THaar2DFeatureList);
  end;

type
  THaar2DLearnerExample = class(TCustomLearnerExample)
  public
    constructor Create(featureList : THaar2DFeatureList; aClassVal : integer);
  end;

implementation

uses Math;

constructor THaar2DFeature.Create(tilted: boolean; x0, y0, w0, h0, wt0, x1, y1, w1, h1, wt1, x2, y2, w2, h2, wt2: integer);
begin
     fRects[0].x := x0;
     fRects[0].y := y0;
     fRects[0].width := w0;
     fRects[0].height := h0;
     fRects[0].weight := wt0;

     fRects[1].x := x1;
     fRects[1].y := y1;
     fRects[1].width := w1;
     fRects[1].height := h1;
     fRects[1].weight := wt1;

     fRects[2].x := x2;
     fRects[2].y := y2;
     fRects[2].width := w2;
     fRects[2].height := h2;
     fRects[2].weight := wt2;

     inherited Create;
end;

constructor THaar2DFeature.Create(tilted: boolean; x0, y0, w0, h0, wt0, x1, y1,
  w1, h1, wt1: integer);
begin
     assert((w0 > 0) and (h0 > 0) and (w1 > 0) and (h1 > 0), 'error cannot add an empty rectangle');
     Create(tilted, x0, y0, w0, h0, wt0, x1, y1, w1, h1, wt1, 0, 0, 0, 0, 0);
end;

function THaar2DFeature.EvalFeature(img : TIntegralImage; x, y: integer; scale: double; colPlane : integer): double;
var w, h : integer;
    i: Integer;
    xOff : integer;
    yOff : integer;
begin
     w := Max(1, Round(fRects[0].width*scale));
     h := Max(1, Round(fRects[0].height*scale));
     xOff := Min(x + Round(fRects[0].x*scale), img.Width - w - 1);
     yOff := Min(y + Round(fRects[0].y*scale), img.Height - h - 1);

     if fIsTilted then
     begin
          Result := fRects[0].weight*Img.TiltRecSum(colPlane, xOff, yoff, w, h);
          for i := 1 to cMaxFeatureRects - 1 do
          begin
               if fRects[i].weight = 0 then
                  break;

               w := Max(1, Round(fRects[i].width*scale));
               h := Max(1, Round(fRects[i].height*scale));
               xOff := Min(x + Round(fRects[i].x*scale), img.Width - w - 1);
               yOff := Min(y + Round(fRects[i].y*scale), img.Height - h - 1);
               Result := Result + fRects[i].weight*Img.TiltRecSum(colPlane, xOff, yOff, w, h);
          end;
     end
     else
     begin
          Result := fRects[0].weight*Img.RecSum(colPlane, x, y, w, h);
          for i := 1 to cMaxFeatureRects - 1 do
          begin
               if fRects[i].weight = 0 then
                  break;

               w := Max(1, Round(fRects[i].width*scale));
               h := Max(1, Round(fRects[i].height*scale));
               xOff := Min(x + Round(fRects[i].x*scale), img.Width - w - 1);
               yOff := Min(y + Round(fRects[i].y*scale), img.Height - h - 1);
               Result := Result + fRects[i].weight*Img.RecSum(colPlane, xOff, yOff, w, h);
          end;
     end;
end;

{ THaar2DFeatureList }

constructor THaar2DFeatureList.Create(img : TIntegralImage; refFeatureList : THaar2DRefFeatureList; numColorPlanes : integer);
begin
     fIntImg := img;
     fNumColorPlanes := numColorPlanes;
     fscale := 1;
     fRefList := refFeatureList;

     if fRefList.FeatureTypes = itSum
     then
         fNormFact := 1
     else
         fNormFact := MaxDouble;

     // calculate the number of features:
     fFeatureVecLen := fRefList.Count*numColorPlanes;
end;

destructor THaar2DFeatureList.Destroy;
begin
     inherited;
end;

function THaar2DFeatureList.GetFeature(index: integer): double;
var mult : double;
    feature : THaar2DFeature;
    w, h : integer;
    colPlane : integer;
    mu : double;
begin
     if index < fRefList.Count
     then
         colPlane := 0
     else
         colPlane := index div fRefList.Count;

     feature := fRefList.Feature(index);
     Result := feature.EvalFeature(fIntImg, fX, fY, fScale, colPlane);

     if SameValue(fNormFact, MaxDouble) then
     begin
          // calc norming factor (according to openCV it seems that the norming factor
          // is calculated only once over the complete image

          // todo: eventually calculate the factor for each feature separately

          // according to wikipedia: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
          // sigma^2 = 1/N* sum( x^2 ) - m^2
          // m ... mean
          // x^2 ... squared image pixels
          //w := Round(fRefList.WindowWidth*fscale);
          //h := Round(fRefList.WindowHeight*fscale);

          w := fIntImg.Width - 4;
          h := fIntImg.Height - 4;

          mult := 1/(w*h);

          //if feature.Tilted then
     //     begin
     //          mu := mult*fIntImg.TiltRecSum(0, fX, fY, w, h);
     //          sigma := cSigmaFact * sqrt(sqr(mu) - mult*fIntImg.TiltRecSumSQR(colPlane, fx, fy, w, h));
     //
     //     end
     //     else
     //     begin
               mu := mult*fIntImg.RecSum(colPlane, 1, 1, w, h);

               fNormFact := 1/(cSigmaFact * sqrt(mult*fIntImg.RecSumSQR(colPlane, 1, 1, w, h) - sqr(mu)));
          //end;
     end;

      // fast light correction
      //Result := (Result - mu)/(cSigmaFact*sigma);

     // note the mean value is not substracted in openCv...
     Result := fNormFact*Result;
end;

procedure THaar2DFeatureList.SetFeature(index: integer; value: double);
begin
     // do nothing here
end;

procedure THaar2DFeatureList.SetFeatureVec(const Feature: array of Double);
begin
     raise Exception.Create('Do not use SetFeatureVec in this class - the creator will do that');
end;

procedure THaar2DFeatureList.SetScale(const Value: double);
begin
     fScale := Value;
end;

procedure THaar2DFeatureList.SetX(const Value: integer);
begin
     if (Value >= 0) and (Value + scale*fRefList.WindowWidth < fIntImg.Width) then
        fX := Value;
end;

procedure THaar2DFeatureList.SetY(const Value: integer);
begin
     if (Value >= 0) and (Value - scale*fRefList.WindowHeight < fIntImg.Height) then
        fY := Value;
end;

{ THaar2DRefFeatureList }

constructor THaar2DRefFeatureList.Create(initWidth, initHeight : integer; numColorPlanes : integer; aFeatureTypes : TIntegralType);
var x: Integer;
    y: Integer;
    dx: Integer;
    dy: Integer;
begin
     inherited Create(True);

     assert((initWidth > 3) and (initHeight > 3), 'Error the frame width/height is wrong');

     fnumColorPlanes := numColorPlanes;
     fWindowWidth := initWidth;
     fWindowHeight := initHeight;
     fFeatureTypes := aFeatureTypes;

     for x := 0 to initWidth - 1 do
     begin
          for y := 0 to initHeight - 1 do
          begin
               for dx := 1 to initWidth - 1 do
               begin
                    for dy := 1 to initHeight - 1 do
                    begin
                         // haar_x2
                         if ( (x+dx*2 <= initWidth) and (y+dy <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( False,
                                                         x, y, dx*2, dy, -1,
                                                         x+dx, y, dx, dy, +2));
                         end;
                         // haar_y2
                         if ( (x+dx <= initWidth) and (y+dy*2 <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x,    y, dx, dy*2, -1,
                                                         x, y+dy, dx, dy,   +2 ) );
                         end;
                         // haar_x3
                         if ( (x+dx*3 <= initWidth) and (y+dy <= initHeight) ) then
                         begin
                             Add(THaar2DFeature.Create( false,
                                                        x,    y, dx*3, dy, -1,
                                                        x+dx, y, dx  , dy, +3 ) );
                         end;
                         // haar_y3
                         if ( (x+dx <= initWidth) and (y+dy*3 <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x, y,    dx, dy*3, -1,
                                                         x, y+dy, dx, dy,   +3 ) );
                         end;

                         // haar_x4
                         if ( (x+dx*4 <= initWidth) and (y+dy <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x,    y, dx*4, dy, -1,
                                                         x+dx, y, dx*2, dy, +2 ) );
                         end;

                         // haar_y4
                         if ( (x+dx <= initWidth ) and (y+dy*4 <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x, y,    dx, dy*4, -1,
                                                         x, y+dy, dx, dy*2, +2 ) );
                         end;

                         // x2_y2
                         if ( (x+dx*2 <= initWidth) and (y+dy*2 <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x,    y,    dx*2, dy*2, -1,
                                                         x,    y,    dx,   dy,   +2,
                                                         x+dx, y+dy, dx,   dy,   +2 ) );
                         end;

                         if ( (x+dx*3 <= initWidth) and (y+dy*3 <= initHeight) ) then
                         begin
                              Add(THaar2DFeature.Create( false,
                                                         x   , y   , dx*3, dy*3, -1,
                                                         x+dx, y+dy, dx  , dy  , +9) );
                         end;

                         if aFeatureTypes = itSumSqrTilted then
                         begin
                              // tilted haar_x2
                              if ( (x+2*dx <= initWidth) and (y+2*dx+dy <= initHeight) and (x-dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x, y, dx*2, dy, -1,
                                                              x, y, dx,   dy, +2 ) );
                              end;
                              // tilted haar_y2
                              if ( (x+dx <= initWidth) and (y+dx+2*dy <= initHeight) and (x-2*dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x, y, dx, 2*dy, -1,
                                                              x, y, dx, dy,   +2 ) );
                              end;
                              // tilted haar_x3
                              if ( (x+3*dx <= initWidth) and (y+3*dx+dy <= initHeight) and (x-dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x,    y,    dx*3, dy, -1,
                                                              x+dx, y+dx, dx,   dy, +3 ) );
                              end;
                              // tilted haar_y3
                              if ( (x+dx <= initWidth) and (y+dx+3*dy <= initHeight) and (x-3*dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x,    y,    dx, 3*dy, -1,
                                                              x-dy, y+dy, dx, dy,   +3 ) );
                              end;
                              // tilted haar_x4
                              if ( (x+4*dx <= initWidth) and (y+4*dx+dy <= initHeight) and (x-dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x,    y,    dx*4, dy, -1,
                                                              x+dx, y+dx, dx*2, dy, +2 ) );
                              end;
                              // tilted haar_y4
                              if ( (x+dx <= initWidth) and (y+dx+4*dy <= initHeight) and (x-4*dy>= 0) ) then
                              begin
                                   Add(THaar2DFeature.Create( true,
                                                              x,    y,    dx, 4*dy, -1,
                                                              x-dy, y+dy, dx, 2*dy, +2 ) );
                              end;
                         end;
                    end;
               end;
          end;
     end;
end;

function THaar2DRefFeatureList.Feature(idx: integer): THaar2DFeature;
begin
     Result := THaar2DFeature(GetItem(idx));
end;

{ THaar2DLearnerExample }

constructor THaar2DLearnerExample.Create(featureList: THaar2DFeatureList;
  aClassVal: integer);
begin
     inherited Create(featureList, True);

     ClassVal := aClassVal;
end;

{ THaar2DExample }

constructor THaar2DExample.Create(featureList: THaar2DFeatureList);
begin
     inherited Create(featureList, False);
end;

end.
