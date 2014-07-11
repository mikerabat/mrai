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

unit Haar2DImageSweep;

interface

uses SysUtils, Image2DSweep, Matrix, Haar2DAdaBoost, Haar2D, CustomBooster;

// ##############################################
// #### Special implementation using the integral image
type
  THaar2DSlidingWindow = class(TObject)
  private
    fWinWidth : integer;
    fWinHeight : integer;
    fIncX, fIncY : integer;
    fScaleInc : double;
    fNumScales : integer;
    fInitScale : double;
    fCl : TCustomBoostingClassifier;

    fRefFeatureList : THaar2DRefFeatureList;
    fCombineRegions: boolean;
  public
    function Classify(img : IMatrix; classVal : integer) : TClassRecList;

    property CombineOverlappingReg : boolean read fCombineRegions write fCombineRegions;

    constructor Create(cl : TCustomBoostingClassifier; xInc, yInc : integer; scaleInc : double; numScales : integer; initScale : double = 1);
    destructor Destroy; override;
  end;


implementation

uses Classes, Integralimg, Math;

{ THaar2DSlidingWindow }

function THaar2DSlidingWindow.Classify(img: IMatrix;
  classVal: integer): TClassRecList;
var iImg : TIntegralImage;
    features : THaar2DFeatureList;
    scaleCnt : integer;
    y : Integer;
    yMax : integer;
    x, xMax : integer;
    example : THaar2DExample;
    props : THaar2DProps;
    c1, c2 : TClassRec;
    intersectWidth, intersectHeight : integer;
    yMin, xMin : integer;
    scale : double;
begin
     if not (fCl.AddObj is THaar2DProps) then
        raise Exception.Create('Error Classifier is not supported');

     props := fCl.AddObj as THaar2DProps;
     Result := TClassRecList.Create;

     iImg := TIntegralImage.Create(Img, Props.FeatureType, Props.NumColorPlanes);

     features := THaar2DFeatureList.Create(iImg, fRefFeatureList, Props.NumColorPlanes);
     try
        example := THaar2DExample.Create(features);
        try
           // ########################################################
           // #### scale loop
           scale := fInitScale - (fNumScales div 2)*fScaleInc;
           for scaleCnt := 0 to fNumScales - 1 do
           begin
                features.Scale := scale;
                // ########################################################
                // #### sweep over the complete image
                yMax := Floor(Img.Height - scale*fWinHeight) - 1;
                xMax := Floor(Img.Width - scale*fWinWidth) - 1;
                y := 0;

                while y < yMax do
                begin
                     x := 0;
                     while x < xMax do
                     begin
                          features.WindowX := x;
                          features.WindowY := y;

                          if fCl.Classify(example) = classVal then
                             Result.Add(TClassRec.Create(x, y, Round(fWinWidth*scale), Round(fWinHeight*scale)));

                          inc(x, fIncX);
                     end;

                     inc(y, fIncY);
                end;

                scale := scale + fScaleInc;
           end;
        finally
               example.Free;
        end;
     finally
            features.Free;
            iImg.Free;
     end;

     // #########################################################
     // #### combine overlapping results

     if fCombineRegions then
     begin
          x := 0;
          while x < Result.Count do
          begin
               c1 := TClassRec(Result[x]);
               for y := Result.Count - 1 downto x + 1 do
               begin
                    // if the overlapping area is more than 75% then combine
                    c2 := TClassRec(Result[y]);

                    yMin := Min(c1.fy + c1.Height, c2.fy + c2.Height);
                    xMin := Min(c1.fx + c1.Width, c2.fx + c2.Width);

                    intersectWidth := Max(0, xMin - Max(c1.x, c2.x));
                    intersectHeight := Max(0, yMin - Max(c1.y, c2.y));

                    if intersectWidth*intersectHeight > 3*c1.Width*c2.Height div 4 then
                    begin
                         c1.fx := (c1.fx + c2.fx) div 2;
                         c1.fy := (c1.fy + c2.fy) div 2;
                         c1.fwidth := (c1.fwidth + c2.fwidth) div 2;
                         c1.fheight := (c1.fheight + c2.fheight) div 2;

                         Result.Delete(y);
                    end;
               end;

               inc(x);
          end;
     end;
end;

constructor THaar2DSlidingWindow.Create(cl: TCustomBoostingClassifier; xInc,
  yInc: integer; scaleInc: double; numScales: integer; initScale : double = 1);
var props : THaar2DProps;
begin
     fCl := cl;
     fIncX := xInc;
     fIncY := yInc;
     fScaleInc := scaleInc;
     fNumScales := numScales;
     fInitScale := initScale;

     if not (fCl.AddObj is THaar2DProps) then
        raise Exception.Create('Error Classifier is not supported');

     props := fCl.AddObj as THaar2DProps;
     fRefFeatureList := props.RefFeatureList;

     fWinWidth := props.WinWidth;
     fWinHeight := props.WinHeight;
     inherited Create;
end;

destructor THaar2DSlidingWindow.Destroy;
begin
     inherited;
end;

end.
