#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3.hpp"

#include <chrono>
using std::placeholders::_1;

double accX;

class Subscriber : public rclcpp::Node
{
  public:
    Subscriber()
    : Node("subscriber")
    {
      subscription_ = this->create_subscription<geometry_msgs::msg::Vector3>(
      "micro_ros_arduino_node_publisher", 10, std::bind(&Subscriber::topic_callback, this, _1));
    }
    void topic_callback(const geometry_msgs::msg::Vector3 msg)
    {
      accX = msg.x;
        // std::cout << msg.x << "\t" << msg.y << "\t" << msg.z << std::endl;
    }
    float getReceivedMessage() const {
      return accX;
    }
  private:
    float accX;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Subscriber>();
    
    torch::jit::script::Module module;
    module = torch::jit::load("/home/poweron/ros2_ws/src/handshake/model/model_dataset4.pt");
    
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor state = torch::randn({1, 1, 1});
    torch::Tensor hn = torch::zeros({2, 1, 8});
    double Duration = 0.02;

    while (rclcpp::ok()){
      std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
      
      accX = node->getReceivedMessage()*9.81;
      
      state[0][0][0] = accX;
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(state);
      inputs.push_back(hn);

      auto output = module.forward(inputs).toTuple();
      at::Tensor output_tensor;
      output_tensor = output->elements()[0].toTensor();
      hn = output->elements()[1].toTensor();
      
      std::cout << output_tensor << std::endl;

      std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
      if (sec.count() < Duration)
        usleep(int((Duration - sec.count())*1000000));
      rclcpp::spin_some(node);
    }
    
    rclcpp::shutdown();
    return 0;
}