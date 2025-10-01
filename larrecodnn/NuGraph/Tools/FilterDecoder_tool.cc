#include "DecoderToolBase.h"

#include "art/Utilities/ToolMacros.h"
#include "canvas/Persistency/Common/Assns.h"
#include "lardataobj/RecoBase/Hit.h"

#include "lardataobj/AnalysisBase/MVAOutput.h"
#include <art/Framework/Principal/Handle.h>
#include <art/Persistency/Common/PtrMaker.h>
#include <torch/torch.h>

using anab::FeatureVector;
using anab::MVADescription;

class FilterDecoder : public DecoderToolBase {
private:
  const art::InputTag fHitLabel;

public:
  /**
   *  @brief  Constructor
   *
   *  @param  pset
   */
  FilterDecoder(const fhicl::ParameterSet& pset);

  /**
   *  @brief  Virtual Destructor
   */
  virtual ~FilterDecoder() noexcept = default;

  /**
   * @brief declareProducts function
   *
   * @param art::ProducesCollector
   */
  void declareProducts(art::ProducesCollector& collector) override
  {
    collector.produces<vector<FeatureVector<1>>>(instancename);
    collector.produces<art::Assns<FeatureVector<1>, recob::Hit>>(instancename);
  }

  /**
   * @brief writeEmptyToEvent function
   *
   * @param art::Event event record
   */
  void writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap) override;

  /**
   * @brief Decoder function
   *
   * @param art::Event event record for decoder
   */
  void writeToEvent(art::Event& e,
                    const vector<vector<size_t>>& idsmap,
                    const vector<NuGraphOutput>& infer_output) override;
};

FilterDecoder::FilterDecoder(const fhicl::ParameterSet& p)
  : DecoderToolBase{p}
  , fHitLabel(p.get<art::InputTag>("HitLabel", "cluster3DCryoE")) {}

void FilterDecoder::writeEmptyToEvent(art::Event& e, const vector<vector<size_t>>& idsmap)
{
  //
  size_t size = 0;
  for (auto& v : idsmap)
    size += v.size();
  auto filtcol =
    std::make_unique<vector<FeatureVector<1>>>(size, FeatureVector<1>(std::array<float, 1>({-1.})));
  auto outputFeatureHitAssns = std::make_unique<art::Assns<FeatureVector<1>, recob::Hit>>();
  e.put(std::move(filtcol), instancename);
  e.put(std::move(outputFeatureHitAssns), instancename);
  //
}

void FilterDecoder::writeToEvent(art::Event& e,
                                 const vector<vector<size_t>>& idsmap,
                                 const vector<NuGraphOutput>& infer_output)
{
  //
  art::ValidHandle<std::vector<recob::Hit>> hitsHandle = e.getValidHandle<std::vector<recob::Hit>>(fHitLabel);
  auto outputFeatureHitAssns = std::make_unique<art::Assns<FeatureVector<1>, recob::Hit>>();
  size_t size = 0;
  for (auto& v : idsmap)
    size += v.size();
  auto filtcol = std::make_unique<vector<FeatureVector<1>>>();
  //
  art::PtrMaker<FeatureVector<1>> fvPtrMaker{e, instancename};

  for (size_t p = 0; p < planes.size(); p++) {
    //
    std::cout << "All ids of plane[" << p << "]:\n";
    for (size_t id : idsmap[p]) {
      std::cout << id << ' ';
    }
    std::cout << '\n';
    const std::vector<float>* x_filter_data = 0;
    for (auto& io : infer_output) {
      if (io.output_name == outputname + planes[p]) x_filter_data = &io.output_vec;
    }
    if (debug) {
      std::cout << "Filter data: " << outputname + planes[p] << std::endl;
      printVector(*x_filter_data);
    }
    //
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    int64_t num_elements = x_filter_data->size();
    const torch::Tensor f =
      torch::from_blob(const_cast<float*>(x_filter_data->data()), {num_elements}, options);
    //
    if (debug) std::cout << "Numel: " << f.numel() << " idsmap[" << p << "]: " << idsmap[p].size() << '\n';
    for (int i = 0; i < f.numel(); ++i) {
      size_t idx = idsmap[p][i];
      std::array<float, 1> input({f[i].item<float>()});
      filtcol->emplace_back(FeatureVector<1>(input));
      const art::Ptr<FeatureVector<1>> fvPtr = fvPtrMaker(filtcol->size()-1);
      const art::Ptr<recob::Hit> hitPtr(hitsHandle, idx);
      std::cout << "Associating FilterVector #" << fvPtr.key() << " with hit #" << hitPtr.key() << '\n';
      outputFeatureHitAssns->addSingle(fvPtr, hitPtr);
    }
  }

  e.put(std::move(filtcol), instancename);
  e.put(std::move(outputFeatureHitAssns), instancename);
}

DEFINE_ART_CLASS_TOOL(FilterDecoder)
